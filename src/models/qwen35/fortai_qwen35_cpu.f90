module fortai_qwen35_cpu
    use, intrinsic :: iso_c_binding, only: c_float, c_int16_t, c_int64_t
    use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real32, real64
    use fortai_gguf_runtime, only: GGML_TYPE_Q8_0, gguf_file_t, gguf_fp16_to_real, gguf_tensor_t
    use fortai_status, only: FORTAI_INVALID, status_t
    implicit none
    private

    interface
        function fortai_float_to_half(value) bind(C, name='fortai_float_to_half') result(bits)
            import c_float, c_int16_t
            real(c_float), value, intent(in) :: value
            integer(c_int16_t) :: bits
        end function fortai_float_to_half

        subroutine fortai_gdn_step(state, key, value, query, decay, beta, head_size, &
                output_scale, output) bind(C, name='fortai_gdn_step')
            import c_float, c_int64_t
            real(c_float), intent(inout) :: state(*)
            real(c_float), intent(in) :: key(*), value(*), query(*)
            real(c_float), value, intent(in) :: decay, beta, output_scale
            integer(c_int64_t), value, intent(in) :: head_size
            real(c_float), intent(out) :: output(*)
        end subroutine fortai_gdn_step

        subroutine fortai_silu(values, count) bind(C, name='fortai_silu')
            import c_float, c_int64_t
            real(c_float), intent(inout) :: values(*)
            integer(c_int64_t), value, intent(in) :: count
        end subroutine fortai_silu

        subroutine fortai_silu_product(left, right, count) bind(C, name='fortai_silu_product')
            import c_float, c_int64_t
            real(c_float), intent(inout) :: left(*)
            real(c_float), intent(in) :: right(*)
            integer(c_int64_t), value, intent(in) :: count
        end subroutine fortai_silu_product
    end interface

    type :: qwen35_cpu_layer_t
        logical :: recurrent = .false.
        integer :: attn_norm = 0
        integer :: post_norm = 0
        integer :: ffn_gate = 0
        integer :: ffn_up = 0
        integer :: ffn_down = 0
        integer :: attn_qkv = 0
        integer :: attn_gate = 0
        integer :: attn_q = 0
        integer :: attn_k = 0
        integer :: attn_v = 0
        integer :: attn_out = 0
        integer :: q_norm = 0
        integer :: k_norm = 0
        integer :: ssm_a = 0
        integer :: ssm_alpha = 0
        integer :: ssm_beta = 0
        integer :: ssm_conv = 0
        integer :: ssm_dt = 0
        integer :: ssm_norm = 0
        integer :: ssm_out = 0
        real(real32), allocatable :: conv_state(:)
        real(real32), allocatable :: gdn_state(:)
        real(real32), allocatable :: key_cache(:)
        real(real32), allocatable :: value_cache(:)
    end type qwen35_cpu_layer_t

    type, public :: qwen35_cpu_model_t
        type(gguf_file_t) :: file
        type(qwen35_cpu_layer_t), allocatable :: layers(:)
        integer(int32) :: hidden_size = 0_int32
        integer(int32) :: vocabulary_size = 0_int32
        integer(int32) :: layer_count = 0_int32
        integer(int32) :: feed_forward_size = 0_int32
        integer(int32) :: attention_heads = 0_int32
        integer(int32) :: attention_heads_kv = 0_int32
        integer(int32) :: attention_head_size = 0_int32
        integer(int32) :: key_length = 0_int32
        integer(int32) :: value_length = 0_int32
        integer(int32) :: rope_dimension = 0_int32
        integer(int32) :: recurrent_head_size = 0_int32
        integer(int32) :: recurrent_key_heads = 0_int32
        integer(int32) :: recurrent_value_heads = 0_int32
        integer(int32) :: recurrent_inner_size = 0_int32
        integer(int32) :: recurrent_conv_size = 0_int32
        integer(int32) :: recurrent_state_size = 0_int32
        integer(int32) :: recurrent_conv_kernel = 0_int32
        integer(int64) :: max_context = 0_int64
        integer(int64) :: bos_token = 0_int64
        integer(int64) :: eos_token = 0_int64
        integer(int32) :: output_norm = 0_int32
        integer(int32) :: output = 0_int32
        integer(int32) :: token_embedding = 0_int32
        real(real32) :: norm_epsilon = 1.0e-6_real32
        real(real32) :: rope_base = 10000000.0_real32
        real(real32), allocatable :: x(:)
        real(real32), allocatable :: residual(:)
        real(real32), allocatable :: normalized(:)
        real(real32), allocatable :: hidden_work(:)
        real(real32), allocatable :: qkv_work(:)
        real(real32), allocatable :: gate_work(:)
        real(real32), allocatable :: q_work(:)
        real(real32), allocatable :: k_work(:)
        real(real32), allocatable :: v_work(:)
        real(real32), allocatable :: attention_work(:)
        real(real32), allocatable :: scores(:)
        real(real32), allocatable :: ffn_gate_work(:)
        real(real32), allocatable :: ffn_up_work(:)
        real(real32), allocatable :: conv_work(:)
        real(real32), allocatable :: beta_work(:)
        real(real32), allocatable :: alpha_work(:)
        real(real32), allocatable :: logits(:)
        integer(int8), allocatable :: quantized_input(:)
        real(real32), allocatable :: quantized_scales(:)
    contains
        procedure :: close => qwen35_cpu_close
        procedure :: forward => qwen35_cpu_forward
        procedure :: gdn_state_add => qwen35_cpu_model_gdn_state_add
        procedure :: gdn_state_value => qwen35_cpu_model_gdn_state_value
        procedure :: layers_key_value => qwen35_cpu_model_layers_key_value
        procedure :: layers_state_store => qwen35_cpu_model_layers_state_store
        procedure :: layers_state_update => qwen35_cpu_model_layers_state_update
        procedure :: open => qwen35_cpu_open
        procedure :: reset => qwen35_cpu_reset
    end type qwen35_cpu_model_t

contains

    subroutine qwen35_cpu_open(self, path, max_context, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        character(len=*), intent(in) :: path
        integer(int64), intent(in), optional :: max_context
        type(status_t), intent(out) :: stat
        integer(int32) :: i, interval
        integer :: work_size

        call self % close()
        call self % file % open(path, stat)
        if (.not. stat % is_ok()) return

        self%hidden_size = int(self%file%meta_int('qwen35.embedding_length', 0_int64), int32)
        self%vocabulary_size = int(self%file%meta_int('qwen35.vocab_size', 0_int64), int32)
        self % layer_count = int(self % file % meta_int('qwen35.block_count', 0_int64), int32)
        self % feed_forward_size = int(self % file % meta_int( &
            'qwen35.feed_forward_length', 0_int64), int32)
        self % attention_heads = int(self % file % meta_int( &
            'qwen35.attention.head_count', 0_int64), int32)
        self % attention_heads_kv = int(self % file % meta_int( &
            'qwen35.attention.head_count_kv', 0_int64), int32)
        self % attention_head_size = int(self % file % meta_int( &
            'qwen35.attention.key_length', 0_int64), int32)
        self % key_length = self % attention_head_size
        self % value_length = int(self % file % meta_int( &
            'qwen35.attention.value_length', 0_int64), int32)
        self % rope_dimension = int(self % file % meta_int( &
            'qwen35.rope.dimension_count', 0_int64), int32)
        self % recurrent_conv_kernel = int(self % file % meta_int( &
            'qwen35.ssm.conv_kernel', 0_int64), int32)
        self % recurrent_state_size = int(self % file % meta_int( &
            'qwen35.ssm.state_size', 0_int64), int32)
        self % recurrent_key_heads = int(self % file % meta_int( &
            'qwen35.ssm.group_count', 0_int64), int32)
        self % recurrent_value_heads = int(self % file % meta_int( &
            'qwen35.ssm.time_step_rank', 0_int64), int32)
        self % recurrent_inner_size = int(self % file % meta_int( &
            'qwen35.ssm.inner_size', 0_int64), int32)
        self % norm_epsilon = self % file % meta_real( &
            'qwen35.attention.layer_norm_rms_epsilon', 1.0e-6_real32)
        self % rope_base = self % file % meta_real('qwen35.rope.freq_base', 10000000.0_real32)
        self % max_context = 256_int64
        if (present(max_context)) self % max_context = max_context
        self % bos_token = self % file % meta_int('tokenizer.ggml.bos_token_id', 0_int64)
        self % eos_token = self % file % meta_int('tokenizer.ggml.eos_token_id', 0_int64)
        interval = int(self % file % meta_int('qwen35.full_attention_interval', 4_int64), int32)
        self % token_embedding = self % file % find_tensor('token_embd.weight')
        self % output_norm = self % file % find_tensor('output_norm.weight')
        self % output = self % file % find_tensor('output.weight')
        if (self % output == 0) self % output = self % token_embedding
        if (self % token_embedding > 0 .and. size(self % file % tensors( &
            self % token_embedding) % shape) == 2) then
            self%hidden_size = int(self%file%tensors(self%token_embedding)%shape(1), int32)
            self%vocabulary_size = int(self%file%tensors(self%token_embedding)%shape(2), int32)
        end if

        if (self%hidden_size <= 0 .or. self%vocabulary_size <= 0 .or. self%layer_count <= 0 &
            .or. self % feed_forward_size <= 0 .or. self % attention_heads <= 0 &
            .or. self % attention_heads_kv <= 0 .or. self % attention_head_size <= 0 &
            .or. self % recurrent_state_size <= 0 .or. self % recurrent_inner_size <= 0 &
            .or. interval <= 0 .or. self % max_context <= 0) then
            call stat % set(FORTAI_INVALID, 'GGUF does not contain a complete Qwen3.5 config')
            return
        end if
        if (mod(self % recurrent_inner_size, self % recurrent_value_heads) /= 0) then
            call stat % set(FORTAI_INVALID, 'Qwen3.5 recurrent inner size is not divisible')
            return
        end if
        if (mod(self % recurrent_value_heads, self % recurrent_key_heads) /= 0 .or. &
            mod(self % attention_heads, self % attention_heads_kv) /= 0) then
            call stat % set(FORTAI_INVALID, 'Qwen3.5 head groups are not divisible')
            return
        end if
        self % recurrent_head_size = self % recurrent_inner_size / self % recurrent_value_heads
        self%recurrent_conv_size = 2 * self%recurrent_state_size * self%recurrent_key_heads &
            + self % recurrent_inner_size

        if (self % token_embedding == 0 .or. self % output_norm == 0) then
            call stat % set(FORTAI_INVALID, 'Qwen3.5 GGUF is missing output tensors')
            return
        end if

        allocate (self % layers(self % layer_count))
        do i = 1, self % layer_count
            self % layers(i) % recurrent = mod(i, interval) /= 0
            call bind_layer(self, i, stat)
            if (.not. stat % is_ok()) return
            if (self % layers(i) % recurrent) then
                allocate (self % layers(i) % conv_state((self % recurrent_conv_kernel - 1) * &
                    self % recurrent_conv_size))
                allocate (self % layers(i) % gdn_state(self % recurrent_head_size * &
                    self % recurrent_head_size * self % recurrent_value_heads))
            else
                allocate (self % layers(i) % key_cache(self % attention_head_size * &
                    self % attention_heads_kv * self % max_context))
                allocate (self % layers(i) % value_cache(self % value_length * &
                    self % attention_heads_kv * self % max_context))
            end if
        end do

        work_size = max(self % feed_forward_size, self % recurrent_conv_size)
        work_size = max(work_size, 2 * self % attention_heads * self % attention_head_size)
        work_size = max(work_size, self % hidden_size)
        allocate (self % x(self % hidden_size), self % residual(self % hidden_size))
        allocate (self % normalized(self % hidden_size), self % hidden_work(self % hidden_size))
        allocate (self % qkv_work(work_size), self % gate_work(work_size))
        allocate (self % q_work(2 * self % attention_heads * self % attention_head_size))
        allocate (self % k_work(self % attention_heads_kv * self % attention_head_size))
        allocate (self % v_work(self % attention_heads_kv * self % value_length))
        allocate (self % attention_work(max(self % attention_heads * self % value_length, &
            self % recurrent_inner_size)))
        allocate (self % scores(self % max_context))
        allocate (self % ffn_gate_work(self % feed_forward_size))
        allocate (self % ffn_up_work(self % feed_forward_size))
        allocate (self % conv_work(self % recurrent_conv_size))
        allocate (self % beta_work(self % recurrent_value_heads))
        allocate (self % alpha_work(self % recurrent_value_heads))
        allocate (self % logits(self % vocabulary_size))
        allocate (self % quantized_input(work_size))
        allocate (self % quantized_scales((work_size + 31) / 32))
        call self % reset()
    end subroutine qwen35_cpu_open

    subroutine qwen35_cpu_close(self)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer :: i

        if (allocated(self % layers)) then
            do i = 1, size(self % layers)
                if (allocated(self % layers(i) % conv_state)) &
                    deallocate (self % layers(i) % conv_state)
                if (allocated(self % layers(i) % gdn_state)) &
                    deallocate (self % layers(i) % gdn_state)
                if (allocated(self % layers(i) % key_cache)) &
                    deallocate (self % layers(i) % key_cache)
                if (allocated(self % layers(i) % value_cache)) &
                    deallocate (self % layers(i) % value_cache)
            end do
            deallocate (self % layers)
        end if
        if (allocated(self % x)) deallocate (self % x)
        if (allocated(self % residual)) deallocate (self % residual)
        if (allocated(self % normalized)) deallocate (self % normalized)
        if (allocated(self % hidden_work)) deallocate (self % hidden_work)
        if (allocated(self % qkv_work)) deallocate (self % qkv_work)
        if (allocated(self % gate_work)) deallocate (self % gate_work)
        if (allocated(self % q_work)) deallocate (self % q_work)
        if (allocated(self % k_work)) deallocate (self % k_work)
        if (allocated(self % v_work)) deallocate (self % v_work)
        if (allocated(self % attention_work)) deallocate (self % attention_work)
        if (allocated(self % scores)) deallocate (self % scores)
        if (allocated(self % ffn_gate_work)) deallocate (self % ffn_gate_work)
        if (allocated(self % ffn_up_work)) deallocate (self % ffn_up_work)
        if (allocated(self % conv_work)) deallocate (self % conv_work)
        if (allocated(self % beta_work)) deallocate (self % beta_work)
        if (allocated(self % alpha_work)) deallocate (self % alpha_work)
        if (allocated(self % logits)) deallocate (self % logits)
        if (allocated(self % quantized_input)) deallocate (self % quantized_input)
        if (allocated(self % quantized_scales)) deallocate (self % quantized_scales)
        call self % file % close()
        self % hidden_size = 0
        self % vocabulary_size = 0
        self % layer_count = 0
    end subroutine qwen35_cpu_close

    subroutine qwen35_cpu_reset(self)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer :: i

        if (allocated(self % x)) self % x = 0.0_real32
        if (allocated(self % layers)) then
            do i = 1, size(self % layers)
                if (allocated(self % layers(i) % conv_state)) self % layers(i) % conv_state = 0.0_real32
                if (allocated(self % layers(i) % gdn_state)) self % layers(i) % gdn_state = 0.0_real32
                if (allocated(self % layers(i) % key_cache)) self % layers(i) % key_cache = 0.0_real32
                if (allocated(self%layers(i)%value_cache)) self%layers(i)%value_cache = 0.0_real32
            end do
        end if
    end subroutine qwen35_cpu_reset

    subroutine qwen35_cpu_forward(self, token_id, position, logits, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id, position
        real(real32), contiguous, intent(out) :: logits(:)
        type(status_t), intent(out) :: stat
        integer :: i

        call stat % clear()
        if (.not. allocated(self % x) .or. size(logits) /= self % vocabulary_size) then
            call stat % set(FORTAI_INVALID, 'Qwen3.5 CPU model is not open')
            return
        end if
        if (token_id < 0_int64 .or. token_id >= self % vocabulary_size) then
            call stat % set(FORTAI_INVALID, 'token id is outside the Qwen3.5 vocabulary')
            return
        end if
        if (position < 0_int64 .or. position >= self % max_context) then
            call stat % set(FORTAI_INVALID, 'Qwen3.5 position exceeds the CPU context')
            return
        end if
        call self%file%tensors(self%token_embedding)%get_row(token_id + 1_int64, self%x, stat)
        if (.not. stat % is_ok()) return

        do i = 1, self % layer_count
            self % residual = self % x
            call rms_norm(self % x, self % file % tensors(self % layers(i) % attn_norm), &
                self % norm_epsilon, self % normalized, stat)
            if (.not. stat % is_ok()) return
            if (self % layers(i) % recurrent) then
                call forward_recurrent(self, self % layers(i), self % normalized, stat)
            else
                call forward_attention(self, self % layers(i), position, stat)
            end if
            if (.not. stat % is_ok()) return
            self % x = self % hidden_work + self % residual
            self % residual = self % x
            call rms_norm(self % x, self % file % tensors(self % layers(i) % post_norm), &
                self % norm_epsilon, self % normalized, stat)
            if (.not. stat % is_ok()) return
            call model_matvec_pair(self, self%layers(i)%ffn_gate, self%layers(i)%ffn_up, &
                self%normalized, self%ffn_gate_work, self%ffn_up_work, stat)
            if (.not. stat % is_ok()) return
            call silu_product(self % ffn_gate_work, self % ffn_up_work)
            call model_matvec(self, self % layers(i) % ffn_down, self % ffn_gate_work, &
                self % hidden_work, stat)
            if (.not. stat % is_ok()) return
            self % x = self % hidden_work + self % residual
        end do

        call rms_norm(self%x, self%file%tensors(self%output_norm), self%norm_epsilon, &
            self % normalized, stat)
        if (.not. stat % is_ok()) return
        call model_matvec(self, self % output, self % normalized, logits, stat)
    end subroutine qwen35_cpu_forward

    subroutine forward_recurrent(self, layer, input, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(qwen35_cpu_layer_t), intent(inout) :: layer
        real(real32), contiguous, intent(in) :: input(:)
        type(status_t), intent(out) :: stat
        integer :: channel, head, j, key_head, slot
        integer :: key_offset, query_offset, state_offset, value_offset
        integer(int64) :: tensor_index
        real(real32) :: accumulator, beta, decay, decay_factor, norm_factor

        call stat % clear()
        call layer_matvec(self, layer%attn_qkv, input, self%qkv_work(1:self%recurrent_conv_size), &
            self % recurrent_conv_size, stat)
        if (.not. stat % is_ok()) return
        call layer_matvec(self, layer%attn_gate, input, self%gate_work(1:self%recurrent_inner_size), &
            self % recurrent_inner_size, stat)
        if (.not. stat % is_ok()) return
        call fortai_silu(self % gate_work, int(self % recurrent_inner_size, c_int64_t))
        call layer_matvec(self, layer % ssm_alpha, input, self % alpha_work, &
            self % recurrent_value_heads, stat)
        if (.not. stat % is_ok()) return
        call layer_matvec(self, layer % ssm_beta, input, self % beta_work, &
            self % recurrent_value_heads, stat)
        if (.not. stat % is_ok()) return

        do channel = 1, self % recurrent_conv_size
            accumulator = 0.0_real32
            do slot = 1, self % recurrent_conv_kernel - 1
                tensor_index = int((channel - 1) * self % recurrent_conv_kernel + slot, int64)
                accumulator = accumulator + self%file%tensors(layer%ssm_conv)%value(tensor_index) &
                    * layer % conv_state((slot - 1) * self % recurrent_conv_size + channel)
            end do
            tensor_index = int((channel - 1) * self % recurrent_conv_kernel + &
                self % recurrent_conv_kernel, int64)
            accumulator = accumulator + self%file%tensors(layer%ssm_conv)%value(tensor_index) &
                * self % qkv_work(channel)
            self % conv_work(channel) = accumulator
        end do
        call fortai_silu(self % conv_work, int(self % recurrent_conv_size, c_int64_t))
        if (self % recurrent_conv_kernel > 2) then
            do slot = 1, self % recurrent_conv_kernel - 2
                layer % conv_state((slot - 1) * self % recurrent_conv_size + 1: &
                    slot * self % recurrent_conv_size) = layer % conv_state( &
                    slot * self % recurrent_conv_size + 1:(slot + 1) * self % recurrent_conv_size)
            end do
        end if
        layer%conv_state((self%recurrent_conv_kernel - 2) * self%recurrent_conv_size + 1: &
            (self % recurrent_conv_kernel - 1) * self % recurrent_conv_size) = self % qkv_work( &
            1:self % recurrent_conv_size)

        do head = 1, self % recurrent_key_heads
            call l2_normalize_slice(self % conv_work, (head - 1) * self % recurrent_head_size + 1, &
                self % recurrent_head_size, self % norm_epsilon)
            call l2_normalize_slice(self % conv_work, self % recurrent_state_size * &
                self % recurrent_key_heads + (head - 1) * self % recurrent_head_size + 1, &
                self % recurrent_head_size, self % norm_epsilon)
        end do
        do head = 1, self % recurrent_value_heads
            key_head = (head - 1) / (self % recurrent_value_heads / self % recurrent_key_heads)
            beta = sigmoid(self % beta_work(head))
            decay = self % file % tensors(layer % ssm_a) % value(int(head, int64)) * &
                softplus(self % alpha_work(head) + self % file % tensors(layer % ssm_dt) % value( &
                int(head, int64)))
            decay_factor = exp(decay)
            state_offset = (head - 1) * self % recurrent_head_size * self % recurrent_head_size
            key_offset = self % recurrent_state_size * self % recurrent_key_heads + &
                key_head * self % recurrent_head_size
            query_offset = key_head * self % recurrent_head_size
            value_offset = 2 * self % recurrent_state_size * self % recurrent_key_heads + &
                (head - 1) * self % recurrent_head_size
            call fortai_gdn_step(layer % gdn_state(state_offset + 1:), &
                self % conv_work(key_offset + 1:), self % conv_work(value_offset + 1:), &
                self % conv_work(query_offset + 1:), decay_factor, beta, &
                int(self % recurrent_head_size, c_int64_t), &
                1.0_real32 / sqrt(real(self % recurrent_head_size, real32)), &
                self % attention_work((head - 1) * self % recurrent_head_size + 1:))
            norm_factor = rms_slice(self % attention_work, &
                (head - 1) * self % recurrent_head_size + 1, self % recurrent_head_size, &
                self % norm_epsilon)
            do j = 1, self % recurrent_head_size
                self % attention_work((head - 1) * self % recurrent_head_size + j) = &
                    self % attention_work((head - 1) * self % recurrent_head_size + j) / norm_factor * &
                    self % file % tensors(layer % ssm_norm) % value(int(j, int64)) * &
                    self % gate_work((head - 1) * self % recurrent_head_size + j)
            end do
        end do
        call model_matvec(self, layer%ssm_out, self%attention_work, self%hidden_work, stat)
    end subroutine forward_recurrent

    subroutine forward_attention(self, layer, position, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(qwen35_cpu_layer_t), intent(inout) :: layer
        integer(int64), intent(in) :: position
        type(status_t), intent(out) :: stat
        integer :: head, i, kv_head, previous, q_offset, k_offset, v_offset
        real(real32) :: score, maximum, normalizer, weight, accumulator

        call stat % clear()
        call model_matvec_triplet(self, layer % attn_q, layer % attn_k, layer % attn_v, &
            self % normalized, self % q_work, self % k_work, self % v_work, stat)
        if (.not. stat % is_ok()) return
        do head = 1, self % attention_heads
            q_offset = (head - 1) * 2 * self % attention_head_size
            call normalize_slice(self % q_work, q_offset + 1, self % attention_head_size, &
                self % norm_epsilon, self % file % tensors(layer % q_norm))
        end do
        do head = 1, self % attention_heads_kv
            k_offset = (head - 1) * self % attention_head_size
            call normalize_slice(self % k_work, k_offset + 1, self % attention_head_size, &
                self % norm_epsilon, self % file % tensors(layer % k_norm))
        end do
        call apply_rope(self % q_work, self % attention_heads, 2 * self % attention_head_size, &
            position, self % rope_dimension, self % rope_base)
        call apply_rope(self % k_work, self % attention_heads_kv, self % attention_head_size, &
            position, self % rope_dimension, self % rope_base)
        call self % layers_state_store(layer, position)

        do head = 1, self % attention_heads
            kv_head = (head - 1) / (self % attention_heads / self % attention_heads_kv)
            maximum = -huge(1.0_real32)
            do previous = 0, int(position)
                score = 0.0_real32
                do i = 1, self % attention_head_size
                    score = score + self % q_work((head - 1) * 2 * self % attention_head_size + i) * &
                        self % layers_key_value(layer, kv_head, previous, i, .true.)
                end do
                self % scores(previous + 1) = score / sqrt(real(self % attention_head_size, real32))
                maximum = max(maximum, self % scores(previous + 1))
            end do
            normalizer = 0.0_real32
            do previous = 0, int(position)
                self % scores(previous + 1) = exp(self % scores(previous + 1) - maximum)
                normalizer = normalizer + self % scores(previous + 1)
            end do
            v_offset = (head - 1) * self % value_length
            do i = 1, self % value_length
                accumulator = 0.0_real32
                do previous = 0, int(position)
                    accumulator = accumulator + self % scores(previous + 1) / normalizer * &
                        self % layers_key_value(layer, kv_head, previous, i, .false.)
                end do
                self % attention_work(v_offset + i) = accumulator * &
                    sigmoid(self % q_work(q_offset_for_gate(head, self % attention_head_size) + i))
            end do
        end do
        call model_matvec(self, layer%attn_out, self%attention_work, self%hidden_work, stat)
    end subroutine forward_attention

    subroutine bind_layer(self, layer_number, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: layer_number
        type(status_t), intent(out) :: stat
        call stat % clear()
        associate (layer => self % layers(layer_number))
            layer % attn_norm = find_layer_tensor(self % file, layer_number - 1, 'attn_norm.weight')
            layer%post_norm = find_layer_tensor(self%file, layer_number - 1, 'post_attention_norm.weight')
            layer % ffn_gate = find_layer_tensor(self % file, layer_number - 1, 'ffn_gate.weight')
            layer % ffn_up = find_layer_tensor(self % file, layer_number - 1, 'ffn_up.weight')
            layer % ffn_down = find_layer_tensor(self % file, layer_number - 1, 'ffn_down.weight')
            if (layer % recurrent) then
                layer % attn_qkv = find_layer_tensor(self % file, layer_number - 1, 'attn_qkv.weight')
                layer % attn_gate = find_layer_tensor(self % file, layer_number - 1, 'attn_gate.weight')
                layer % ssm_a = find_layer_tensor(self % file, layer_number - 1, 'ssm_a')
                layer % ssm_alpha = find_layer_tensor(self % file, layer_number - 1, 'ssm_alpha.weight')
                layer % ssm_beta = find_layer_tensor(self % file, layer_number - 1, 'ssm_beta.weight')
                layer % ssm_conv = find_layer_tensor(self % file, layer_number - 1, 'ssm_conv1d.weight')
                layer % ssm_dt = find_layer_tensor(self % file, layer_number - 1, 'ssm_dt.bias')
                layer % ssm_norm = find_layer_tensor(self % file, layer_number - 1, 'ssm_norm.weight')
                layer % ssm_out = find_layer_tensor(self % file, layer_number - 1, 'ssm_out.weight')
            else
                layer % attn_q = find_layer_tensor(self % file, layer_number - 1, 'attn_q.weight')
                layer % attn_k = find_layer_tensor(self % file, layer_number - 1, 'attn_k.weight')
                layer % attn_v = find_layer_tensor(self % file, layer_number - 1, 'attn_v.weight')
                layer%attn_out = find_layer_tensor(self%file, layer_number - 1, 'attn_output.weight')
                layer % q_norm = find_layer_tensor(self % file, layer_number - 1, 'attn_q_norm.weight')
                layer % k_norm = find_layer_tensor(self % file, layer_number - 1, 'attn_k_norm.weight')
            end if
            if (layer % attn_norm == 0 .or. layer % post_norm == 0 .or. layer % ffn_gate == 0 &
                .or. layer % ffn_up == 0 .or. layer % ffn_down == 0) then
                call stat % set(FORTAI_INVALID, 'Qwen3.5 layer is missing an FFN tensor')
                return
            end if
            if (layer % recurrent) then
                if (layer % attn_qkv == 0 .or. layer % attn_gate == 0 .or. layer % ssm_a == 0 &
                    .or. layer % ssm_alpha == 0 .or. layer % ssm_beta == 0 .or. layer % ssm_conv == 0 &
                    .or. layer % ssm_dt == 0 .or. layer % ssm_norm == 0 .or. layer % ssm_out == 0) then
                    call stat % set(FORTAI_INVALID, 'Qwen3.5 recurrent layer is incomplete')
                end if
            else if (layer % attn_q == 0 .or. layer % attn_k == 0 .or. layer % attn_v == 0 &
                    .or. layer % attn_out == 0 .or. layer % q_norm == 0 .or. layer % k_norm == 0) then
                call stat % set(FORTAI_INVALID, 'Qwen3.5 attention layer is incomplete')
            end if
        end associate
    end subroutine bind_layer

    integer function find_layer_tensor(file, layer, suffix)
        type(gguf_file_t), intent(in) :: file
        integer, intent(in) :: layer
        character(len=*), intent(in) :: suffix
        character(len=128) :: name

        write (name, '("blk.", i0, ".", a)') layer, suffix
        find_layer_tensor = file % find_tensor(trim(name))
    end function find_layer_tensor

    subroutine layer_matvec(self, tensor_index, input, output, expected, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: tensor_index, expected
        real(real32), contiguous, intent(in) :: input(:)
        real(real32), contiguous, intent(out) :: output(:)
        type(status_t), intent(out) :: stat

        call stat % clear()
        if (tensor_index == 0 .or. size(output) /= expected) then
            call stat % set(FORTAI_INVALID, 'Qwen3.5 CPU tensor binding is invalid')
            return
        end if
        call model_matvec(self, tensor_index, input, output, stat)
    end subroutine layer_matvec

    subroutine model_matvec(self, tensor_index, input, output, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: tensor_index
        real(real32), contiguous, intent(in) :: input(:)
        real(real32), contiguous, intent(out) :: output(:)
        type(status_t), intent(out) :: stat

        call stat % clear()
        if (tensor_index == 0 .or. size(input) > size(self % quantized_input)) then
            call stat % set(FORTAI_INVALID, 'Qwen3.5 CPU matvec workspace is invalid')
            return
        end if
        if (self % file % tensors(tensor_index) % value_type == GGML_TYPE_Q8_0) then
            call self % file % tensors(tensor_index) % matvec_q8(input, output, &
                self % quantized_input, self % quantized_scales, stat)
        else
            call self % file % tensors(tensor_index) % matvec(input, output, stat)
        end if
    end subroutine model_matvec

    subroutine model_matvec_pair(self, first_index, second_index, input, first_output, &
            second_output, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: first_index, second_index
        real(real32), contiguous, intent(in) :: input(:)
        real(real32), contiguous, intent(out) :: first_output(:), second_output(:)
        type(status_t), intent(out) :: stat

        call stat%clear()
        if (first_index == 0 .or. second_index == 0) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CPU paired tensor binding is invalid')
            return
        end if
        if (self%file%tensors(first_index)%value_type == GGML_TYPE_Q8_0 .and. &
            self%file%tensors(second_index)%value_type == GGML_TYPE_Q8_0) then
            call self%file%tensors(first_index)%matvec_pair_q8( &
                self%file%tensors(second_index), input, first_output, second_output, &
                self%quantized_input, self%quantized_scales, stat)
        else
            call model_matvec(self, first_index, input, first_output, stat)
            if (.not. stat%is_ok()) return
            call model_matvec(self, second_index, input, second_output, stat)
        end if
    end subroutine model_matvec_pair

    subroutine model_matvec_triplet(self, first_index, second_index, third_index, input, &
            first_output, second_output, third_output, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: first_index, second_index, third_index
        real(real32), contiguous, intent(in) :: input(:)
        real(real32), contiguous, intent(out) :: first_output(:), second_output(:), third_output(:)
        type(status_t), intent(out) :: stat

        call stat%clear()
        if (first_index == 0 .or. second_index == 0 .or. third_index == 0) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CPU triplet tensor binding is invalid')
            return
        end if
        if (self%file%tensors(first_index)%value_type == GGML_TYPE_Q8_0 .and. &
            self%file%tensors(second_index)%value_type == GGML_TYPE_Q8_0 .and. &
            self%file%tensors(third_index)%value_type == GGML_TYPE_Q8_0) then
            call self%file%tensors(first_index)%matvec_triplet_q8( &
                self%file%tensors(second_index), self%file%tensors(third_index), input, &
                first_output, second_output, third_output, self%quantized_input, &
                self%quantized_scales, stat)
        else
            call model_matvec(self, first_index, input, first_output, stat)
            if (.not. stat%is_ok()) return
            call model_matvec(self, second_index, input, second_output, stat)
            if (.not. stat%is_ok()) return
            call model_matvec(self, third_index, input, third_output, stat)
        end if
    end subroutine model_matvec_triplet

    subroutine rms_norm(input, weights, epsilon, output, stat)
        real(real32), intent(in) :: input(:), epsilon
        type(gguf_tensor_t), intent(in) :: weights
        real(real32), intent(out) :: output(:)
        type(status_t), intent(out) :: stat
        integer :: i
        real(real64) :: sum_squares
        real(real32) :: mean, inverse_scale

        call stat % clear()
        if (size(output) /= size(input) .or. size(weights % shape) /= 1) then
            call stat % set(FORTAI_INVALID, 'RMS norm dimensions do not agree')
            return
        end if
        sum_squares = 0.0_real64
        do i = 1, size(input)
            sum_squares = sum_squares + real(input(i), real64) * real(input(i), real64)
        end do
        mean = real(sum_squares / real(size(input), real64), real32)
        inverse_scale = 1.0_real32 / sqrt(mean + epsilon)
        do i = 1, size(input)
            output(i) = input(i) * inverse_scale
            output(i) = output(i) * weights % value(int(i, int64))
        end do
    end subroutine rms_norm

    subroutine normalize_slice(values, first, length, epsilon, weights)
        real(real32), intent(inout) :: values(:)
        integer, intent(in) :: first, length
        real(real32), intent(in) :: epsilon
        type(gguf_tensor_t), intent(in) :: weights
        integer :: i
        real(real64) :: sum_squares
        real(real32) :: mean, inverse_scale

        sum_squares = 0.0_real64
        do i = first, first + length - 1
            sum_squares = sum_squares + real(values(i), real64) * real(values(i), real64)
        end do
        mean = real(sum_squares / real(length, real64), real32)
        inverse_scale = 1.0_real32 / sqrt(mean + epsilon)
        do i = 0, length - 1
            values(first + i) = values(first + i) * inverse_scale
            values(first + i) = values(first + i) * weights % value(int(i + 1, int64))
        end do
    end subroutine normalize_slice

    subroutine l2_normalize_slice(values, first, length, epsilon)
        real(real32), intent(inout) :: values(:)
        integer, intent(in) :: first, length
        real(real32), intent(in) :: epsilon
        integer :: i
        real(real64) :: sum_squares
        real(real32) :: inverse_scale

        sum_squares = 0.0_real64
        do i = first, first + length - 1
            sum_squares = sum_squares + real(values(i), real64) * real(values(i), real64)
        end do
        inverse_scale = 1.0_real32 / max(sqrt(real(sum_squares, real32)), epsilon)
        do i = first, first + length - 1
            values(i) = values(i) * inverse_scale
        end do
    end subroutine l2_normalize_slice

    subroutine apply_rope(values, heads, stride, position, dimension, base)
        real(real32), intent(inout) :: values(:)
        integer, intent(in) :: heads, stride, dimension
        integer(int64), intent(in) :: position
        real(real32), intent(in) :: base
        integer :: head, i, half_dimension
        real(real32) :: angle, cosine, sine, first, second

        ! Qwen3.5 uses interleaved MRoPE with NeoX-style half-vector pairs.
        ! For text all three position streams are equal, so the frequency
        ! sequence is the usual one; vision will require explicit sections.
        half_dimension = dimension / 2
        do head = 0, heads - 1
            do i = 0, half_dimension - 1
                angle = real(position, real32) / base**(real(2 * i, real32) / &
                    real(dimension, real32))
                cosine = cos(angle)
                sine = sin(angle)
                first = values(head * stride + i + 1)
                second = values(head * stride + i + half_dimension + 1)
                values(head * stride + i + 1) = first * cosine - second * sine
                values(head * stride + i + half_dimension + 1) = first * sine + second * cosine
            end do
        end do
    end subroutine apply_rope

    subroutine silu_product(left, right)
        real(real32), contiguous, intent(inout) :: left(:)
        real(real32), contiguous, intent(in) :: right(:)

        call fortai_silu_product(left, right, int(size(left), c_int64_t))
    end subroutine silu_product

    real(real32) function sigmoid(value)
        real(real32), intent(in) :: value

        if (value >= 0.0_real32) then
            sigmoid = 1.0_real32 / (1.0_real32 + exp(-value))
        else
            sigmoid = exp(value) / (1.0_real32 + exp(value))
        end if
    end function sigmoid

    real(real32) function softplus(value)
        real(real32), intent(in) :: value

        if (value > 20.0_real32) then
            softplus = value
        else if (value < -20.0_real32) then
            softplus = exp(value)
        else
            softplus = log(1.0_real32 + exp(value))
        end if
    end function softplus

    real(real32) function rms_slice(values, first, length, epsilon)
        real(real32), intent(in) :: values(:), epsilon
        integer, intent(in) :: first, length
        integer :: i
        real(real64) :: sum_squares

        sum_squares = 0.0_real64
        do i = first, first + length - 1
            sum_squares = sum_squares + real(values(i), real64) * real(values(i), real64)
        end do
        rms_slice = sqrt(real(sum_squares / real(length, real64), real32) + epsilon)
    end function rms_slice

    integer function q_offset_for_gate(head, head_size)
        integer, intent(in) :: head, head_size

        ! Return the zero-based offset so the caller's one-based i selects
        ! the first gate value at the start of the second Q/G half.
        q_offset_for_gate = (head - 1) * 2 * head_size + head_size
    end function q_offset_for_gate

    subroutine qwen35_cpu_model_layers_state_update(self, layer, head, row, column, decay)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(qwen35_cpu_layer_t), intent(inout) :: layer
        integer, intent(in) :: head, row, column
        real(real32), intent(in) :: decay
        integer :: index

        index = (head - 1) * self % recurrent_head_size * self % recurrent_head_size + &
            (row - 1) * self % recurrent_head_size + column
        layer % gdn_state(index) = layer % gdn_state(index) * exp(decay)
    end subroutine qwen35_cpu_model_layers_state_update

    real(real32) function qwen35_cpu_model_gdn_state_value(self, layer, head, row, column)
        class(qwen35_cpu_model_t), intent(in) :: self
        type(qwen35_cpu_layer_t), intent(in) :: layer
        integer, intent(in) :: head, row, column
        integer :: index

        index = (head - 1) * self % recurrent_head_size * self % recurrent_head_size + &
            (row - 1) * self % recurrent_head_size + column
        qwen35_cpu_model_gdn_state_value = layer % gdn_state(index)
    end function qwen35_cpu_model_gdn_state_value

    subroutine qwen35_cpu_model_gdn_state_add(self, layer, head, row, column, value)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(qwen35_cpu_layer_t), intent(inout) :: layer
        integer, intent(in) :: head, row, column
        real(real32), intent(in) :: value
        integer :: index

        index = (head - 1) * self % recurrent_head_size * self % recurrent_head_size + &
            (row - 1) * self % recurrent_head_size + column
        layer % gdn_state(index) = layer % gdn_state(index) + value
    end subroutine qwen35_cpu_model_gdn_state_add

    subroutine qwen35_cpu_model_layers_state_store(self, layer, position)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(qwen35_cpu_layer_t), intent(inout) :: layer
        integer(int64), intent(in) :: position
        integer :: i, head, offset

        do head = 1, self % attention_heads_kv
            offset = int(position) * self % attention_head_size * self % attention_heads_kv + &
                (head - 1) * self % attention_head_size
            do i = 1, self % attention_head_size
                layer%key_cache(offset + i) = gguf_fp16_to_real(fortai_float_to_half( &
                    self%k_work((head - 1) * self%attention_head_size + i)))
            end do
            offset = int(position) * self % value_length * self % attention_heads_kv + &
                (head - 1) * self % value_length
            do i = 1, self % value_length
                layer % value_cache(offset + i) = gguf_fp16_to_real(fortai_float_to_half( &
                    self % v_work((head - 1) * self % value_length + i)))
            end do
        end do
    end subroutine qwen35_cpu_model_layers_state_store

    real(real32) function qwen35_cpu_model_layers_key_value(self, layer, head, position, index, key)
        class(qwen35_cpu_model_t), intent(in) :: self
        type(qwen35_cpu_layer_t), intent(in) :: layer
        integer, intent(in) :: head, index
        integer, intent(in) :: position
        logical, intent(in) :: key
        integer :: offset

        if (key) then
            offset = position * self % attention_head_size * self % attention_heads_kv + &
                head * self % attention_head_size + index
            qwen35_cpu_model_layers_key_value = layer % key_cache(offset)
        else
            offset = position * self % value_length * self % attention_heads_kv + &
                head * self % value_length + index
            qwen35_cpu_model_layers_key_value = layer % value_cache(offset)
        end if
    end function qwen35_cpu_model_layers_key_value

end module fortai_qwen35_cpu
