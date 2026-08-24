module fortai_qwen35_cpu
    use, intrinsic :: iso_c_binding, only: c_associated, c_float, c_int, c_int16_t, c_int64_t, c_int8_t, &
        c_null_ptr, c_ptr, c_size_t
    use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real32, real64
    use fortai_backend_cuda, only: cuda_q8_context_t, cuda_q8_matvec_host, &
        cuda_q8_ffn_host, cuda_q8_ffn_device, cuda_q8_matvec_host_pair, &
        cuda_q8_matvec_host_triplet, cuda_q8_matvec_host_triplet_contiguous, cuda_q8_weights_t, &
        cuda_q8_matvec_device_f32, cuda_qwen35_add_device, cuda_qwen35_copy_device, &
        cuda_qwen35_embedding_device, cuda_qwen35_rms_norm_device
    use fortai_backend_cuda, only: cuda_q4_context_t, cuda_q4_weights_t, cuda_q4_matvec_host, &
        cuda_q4_matvec_host_pair, cuda_q4_matvec_host_triplet
    use fortai_backend_cuda, only: cuda_qwen35_attention_t, cuda_qwen35_recurrent_t
    use fortai_gguf_runtime, only: GGML_TYPE_Q8_0, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, &
        GGML_TYPE_Q6_K, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS, gguf_file_t, &
        fortai_ggml_quant_matvec_pair, fortai_ggml_quant_matvec_triplet, gguf_fp16_to_real, &
        gguf_quant_cache_clear, gguf_tensor_t
    use fortai_status, only: FORTAI_INVALID, FORTAI_UNSUPPORTED, status_t
    implicit none
    private

    interface
        function fortai_float_to_half(value) bind(C, name='fortai_float_to_half') result(bits)
            import c_float, c_int16_t
            real(c_float), value, intent(in) :: value
            integer(c_int16_t) :: bits
        end function fortai_float_to_half

        subroutine fortai_q8_quantize(vector, quantized, scales, count) &
                bind(C, name='fortai_q8_quantize')
            import c_float, c_int8_t, c_int64_t
            real(c_float), intent(in) :: vector(*)
            integer(c_int8_t), intent(out) :: quantized(*)
            real(c_float), intent(out) :: scales(*)
            integer(c_int64_t), value, intent(in) :: count
        end subroutine fortai_q8_quantize

        subroutine fortai_flash_attention_f16(query, key_cache, value_cache, count, &
                key_stride, value_stride, key_size, value_size, scale, output) &
                bind(C, name='fortai_flash_attention_f16')
            import c_float, c_int64_t
            real(c_float), intent(in) :: query(*), key_cache(*), value_cache(*)
            integer(c_int64_t), value, intent(in) :: count, key_stride, value_stride
            integer(c_int64_t), value, intent(in) :: key_size, value_size
            real(c_float), value, intent(in) :: scale
            real(c_float), intent(out) :: output(*)
        end subroutine fortai_flash_attention_f16

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
        type(cuda_qwen35_recurrent_t) :: cuda_recurrent
        type(cuda_qwen35_attention_t) :: cuda_attention
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
        real(real32), allocatable :: qkv_download_work(:)
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
        type(cuda_q8_context_t) :: cuda
        type(cuda_q8_weights_t), allocatable :: cuda_weights(:)
        type(cuda_q4_context_t) :: cuda_q4
        type(cuda_q4_weights_t), allocatable :: cuda_q4_weights(:)
        type(c_ptr), allocatable :: cuda_attn_norm(:)
        type(c_ptr), allocatable :: cuda_post_norm(:)
        type(c_ptr) :: cuda_x = c_null_ptr
        type(c_ptr) :: cuda_residual = c_null_ptr
        type(c_ptr) :: cuda_normalized = c_null_ptr
        type(c_ptr) :: cuda_hidden = c_null_ptr
        type(c_ptr) :: cuda_output_norm = c_null_ptr
        type(c_ptr) :: cuda_logits = c_null_ptr
        logical :: cuda_enabled = .false.
        logical :: cuda_device_pipeline = .false.
        logical :: cuda_graph_enabled = .false.
        logical :: cuda_graph_ready = .false.
        logical :: persistent_openmp = .false.
        logical :: persistent_openmp_active = .false.
    contains
        procedure :: close => qwen35_cpu_close
        procedure :: enable_cuda => qwen35_cpu_enable_cuda
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
        ! Qwen3.8-27B Q4_K_XL carries one MTP/nextn block after the base
        ! transformer.  llama.cpp intentionally leaves blk.64.* unused;
        ! keep the base model's 64 layers in the native runner as well.
        if (self % layer_count > 64 .and. self % file % find_tensor( &
            'blk.64.nextn.eh_proj.weight') > 0) self % layer_count = 64
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
        self%persistent_openmp = .false.
        call enable_persistent_openmp(self)
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
        call check_tensor_shape(self, self % token_embedding, 2, self % hidden_size, &
            self % vocabulary_size, stat)
        if (.not. stat % is_ok()) return
        call check_tensor_shape(self, self % output_norm, 1, self % hidden_size, 0, stat)
        if (.not. stat % is_ok()) return
        call check_tensor_shape(self, self % output, 2, self % hidden_size, &
            self % vocabulary_size, stat)
        if (.not. stat % is_ok()) return

        allocate (self % layers(self % layer_count))
        do i = 1, self % layer_count
            ! The published Qwen3.8-27B GGUF has one terminal full-attention
            ! block (blk.64) in addition to the regular interval.  Derive
            ! the layer kind from the tensor schema so that this extra block
            ! is bound as attention instead of being rejected as incomplete
            ! recurrent state.
            self % layers(i) % recurrent = find_layer_tensor(self % file, i - 1, 'attn_q.weight') == 0
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
        allocate (self % qkv_download_work(size(self % q_work) + size(self % k_work) + size(self % v_work)))
        allocate (self % attention_work(max(self % attention_heads * self % value_length, &
            self % recurrent_inner_size)))
        allocate (self % scores(self % max_context))
        allocate (self % ffn_gate_work(self % feed_forward_size))
        allocate (self % ffn_up_work(self % feed_forward_size))
        allocate (self % conv_work(self % recurrent_conv_size))
        allocate (self % beta_work(self % recurrent_value_heads))
        allocate (self % alpha_work(self % recurrent_value_heads))
        allocate (self % logits(self % vocabulary_size))
        allocate (self % quantized_input(work_size + 2 * ((work_size + 31) / 32)))
        allocate (self % quantized_scales((work_size + 31) / 32))
        call self % reset()
    end subroutine qwen35_cpu_open

    subroutine qwen35_cpu_enable_cuda(self, device, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: device
        type(status_t), intent(out) :: stat
        type(status_t) :: cleanup_stat
        integer :: i, j, rows, width, q4_device, q4_second_device
        integer(int64) :: q4_bytes(2)
        logical :: have_q4

        call stat%clear()
        if (.not. allocated(self%file%tensors)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA model is not open')
            return
        end if
        if (allocated(self%cuda_weights)) then
            call cuda_device_pipeline_cleanup(self, cleanup_stat)
            if (allocated(self%layers)) then
                do i = 1, size(self%layers)
                    call self%layers(i)%cuda_recurrent%destroy(cleanup_stat)
                    call self%layers(i)%cuda_attention%destroy(cleanup_stat)
                end do
            end if
            do i = 1, size(self%cuda_weights)
                call self%cuda_weights(i)%destroy(cleanup_stat)
            end do
            deallocate(self%cuda_weights)
        end if
        if (allocated(self%cuda_q4_weights)) then
            do i = 1, size(self%cuda_q4_weights)
                call self%cuda_q4_weights(i)%destroy(cleanup_stat)
            end do
            deallocate(self%cuda_q4_weights)
        end if
        call self%cuda_q4%destroy(cleanup_stat)
        self%cuda_enabled = .false.
        call self%cuda%destroy(cleanup_stat)
        call self%cuda%create(device, stat)
        if (.not. stat%is_ok()) return

        allocate(self%cuda_weights(size(self%file%tensors)))
        do i = 1, size(self%file%tensors)
            if (self%file%tensors(i)%value_type /= GGML_TYPE_Q8_0) cycle
            if (size(self%file%tensors(i)%shape) /= 2) cycle
            width = int(self%file%tensors(i)%shape(1))
            rows = int(self%file%tensors(i)%shape(2))
            call self%cuda_weights(i)%upload(self%cuda, self%file%tensors(i)%bytes, &
                int(size(self%file%tensors(i)%bytes), c_size_t), rows, width, stat)
            if (.not. stat%is_ok()) then
                do j = 1, i - 1
                    call self%cuda_weights(j)%destroy(cleanup_stat)
                end do
                deallocate(self%cuda_weights)
                call self%cuda%destroy(cleanup_stat)
                return
            end if
        end do
        have_q4 = .false.
        do i = 1, size(self%file%tensors)
            if (is_q4_xl_type(self%file%tensors(i)%value_type) .and. &
                size(self%file%tensors(i)%shape) == 2) then
                have_q4 = .true.
                exit
            end if
        end do
        if (have_q4) then
            q4_second_device = device + 1
            ! The default pair is device/device+1; an explicit override is
            ! parsed below without making the Q8 single-device path depend on it.
            block
                character(len=32) :: second_text
                integer :: second_length, second_status
                call get_environment_variable('FORTAI_CUDA_Q4_SECOND_DEVICE', second_text, length=second_length)
                if (second_length > 0) read(second_text(1:second_length), *, iostat=second_status) q4_second_device
            end block
            call self%cuda_q4%create(device, q4_second_device, stat)
            if (.not. stat%is_ok()) then
                do j = 1, size(self%cuda_weights)
                    call self%cuda_weights(j)%destroy(cleanup_stat)
                end do
                deallocate(self%cuda_weights)
                call self%cuda%destroy(cleanup_stat)
                return
            end if
            allocate(self%cuda_q4_weights(size(self%file%tensors)))
            q4_bytes = 0_int64
            do i = 1, size(self%file%tensors)
                if (.not. is_q4_xl_type(self%file%tensors(i)%value_type)) cycle
                if (size(self%file%tensors(i)%shape) /= 2) cycle
                if (is_unused_q4_tensor(self, i)) cycle
                q4_device = 1
                if (q4_bytes(1) <= q4_bytes(2)) q4_device = 0
                rows = int(self%file%tensors(i)%shape(2))
                width = int(self%file%tensors(i)%shape(1))
                call self%cuda_q4_weights(i)%upload(self%cuda_q4, self%file%tensors(i)%value_type, &
                    self%file%tensors(i)%bytes, int(size(self%file%tensors(i)%bytes), c_size_t), &
                    rows, width, q4_device, stat)
                if (.not. stat%is_ok()) then
                    do j = 1, size(self%cuda_q4_weights)
                        call self%cuda_q4_weights(j)%destroy(cleanup_stat)
                    end do
                    deallocate(self%cuda_q4_weights)
                    call self%cuda_q4%destroy(cleanup_stat)
                    do j = 1, size(self%cuda_weights)
                        call self%cuda_weights(j)%destroy(cleanup_stat)
                    end do
                    deallocate(self%cuda_weights)
                    call self%cuda%destroy(cleanup_stat)
                    return
                end if
                q4_bytes(q4_device + 1) = q4_bytes(q4_device + 1) + &
                    int(size(self%file%tensors(i)%bytes), int64)
            end do
            call self%cuda_q4%synchronize(stat)
            if (.not. stat%is_ok()) then
                do j = 1, size(self%cuda_q4_weights)
                    call self%cuda_q4_weights(j)%destroy(cleanup_stat)
                end do
                deallocate(self%cuda_q4_weights)
                call self%cuda_q4%destroy(cleanup_stat)
                do j = 1, size(self%cuda_weights)
                    call self%cuda_weights(j)%destroy(cleanup_stat)
                end do
                deallocate(self%cuda_weights)
                call self%cuda%destroy(cleanup_stat)
                return
            end if
        end if
        do i = 1, size(self%layers)
            if (.not. self%layers(i)%recurrent) cycle
            if (.not. all_q8_recurrent_weights(self, i)) cycle
            call self%layers(i)%cuda_recurrent%create(self%cuda, &
                self%cuda_weights(self%layers(i)%attn_qkv), self%cuda_weights(self%layers(i)%attn_gate), &
                self%cuda_weights(self%layers(i)%ssm_alpha), self%cuda_weights(self%layers(i)%ssm_beta), &
                self%cuda_weights(self%layers(i)%ssm_out), self%file%tensors(self%layers(i)%ssm_conv)%bytes, &
                int(size(self%file%tensors(self%layers(i)%ssm_conv)%bytes), c_size_t), &
                self%recurrent_conv_size, self%recurrent_conv_kernel, &
                self%file%tensors(self%layers(i)%ssm_a)%bytes, &
                int(size(self%file%tensors(self%layers(i)%ssm_a)%bytes), c_size_t), &
                self%file%tensors(self%layers(i)%ssm_dt)%bytes, &
                int(size(self%file%tensors(self%layers(i)%ssm_dt)%bytes), c_size_t), &
                self%file%tensors(self%layers(i)%ssm_norm)%bytes, &
                int(size(self%file%tensors(self%layers(i)%ssm_norm)%bytes), c_size_t), &
                self%recurrent_state_size, self%recurrent_key_heads, self%recurrent_value_heads, &
                self%recurrent_head_size, self%recurrent_inner_size, self%norm_epsilon, stat)
            if (.not. stat%is_ok()) then
                do j = 1, size(self%layers)
                    call self%layers(j)%cuda_recurrent%destroy(cleanup_stat)
                    call self%layers(j)%cuda_attention%destroy(cleanup_stat)
                end do
                do j = 1, size(self%cuda_weights)
                    call self%cuda_weights(j)%destroy(cleanup_stat)
                end do
                deallocate(self%cuda_weights)
                call self%cuda%destroy(cleanup_stat)
                return
            end if
        end do
        do i = 1, size(self%layers)
            if (self%layers(i)%recurrent) cycle
            if (.not. all_q8_attention_weights(self, i)) cycle
            if (size(self%file%tensors(self%layers(i)%q_norm)%bytes) /= &
                self%attention_head_size * storage_size(self%x(1)) / 8) cycle
            if (size(self%file%tensors(self%layers(i)%k_norm)%bytes) /= &
                self%attention_head_size * storage_size(self%x(1)) / 8) cycle
            call self%layers(i)%cuda_attention%create(self%cuda, &
                self%cuda_weights(self%layers(i)%attn_q), self%cuda_weights(self%layers(i)%attn_k), &
                self%cuda_weights(self%layers(i)%attn_v), self%cuda_weights(self%layers(i)%attn_out), &
                self%file%tensors(self%layers(i)%q_norm)%bytes, &
                int(size(self%file%tensors(self%layers(i)%q_norm)%bytes), c_size_t), &
                self%file%tensors(self%layers(i)%k_norm)%bytes, &
                int(size(self%file%tensors(self%layers(i)%k_norm)%bytes), c_size_t), &
                self%attention_heads, self%attention_heads_kv, self%attention_head_size, &
                self%value_length, int(self%max_context), self%rope_dimension, self%rope_base, &
                self%norm_epsilon, cleanup_stat)
            if (.not. cleanup_stat%is_ok()) call cleanup_stat%clear()
        end do
        self%cuda_enabled = .true.
        call setup_cuda_device_pipeline(self, cleanup_stat)
        if (.not. cleanup_stat%is_ok()) call cleanup_stat%clear()
    end subroutine qwen35_cpu_enable_cuda

    subroutine setup_cuda_device_pipeline(self, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(status_t), intent(out) :: stat
        integer :: i, graph_length, pipeline_length
        integer(c_size_t) :: hidden_bytes
        character(len=8) :: graph_env
        character(len=8) :: pipeline_env

        call stat%clear()
        self%cuda_device_pipeline = .false.
        self%cuda_graph_enabled = .false.
        self%cuda_graph_ready = .false.
        call get_environment_variable('FORTAI_DISABLE_CUDA_DEVICE_PIPELINE', pipeline_env, length=pipeline_length)
        if (pipeline_length > 0) then
            if (pipeline_env(1:1) == '1') return
        end if
        call get_environment_variable('FORTAI_ENABLE_CUDA_GRAPH', graph_env, length=graph_length)
        if (graph_length > 0) then
            if (graph_env(1:1) == '1') self%cuda_graph_enabled = .true.
        end if
        call get_environment_variable('FORTAI_DISABLE_CUDA_GRAPH', graph_env, length=graph_length)
        if (graph_length > 0) then
            if (graph_env(1:graph_length) == '1') self%cuda_graph_enabled = .false.
        end if
        if (.not. self%cuda_enabled) return
        if (.not. allocated(self%layers)) then
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA device pipeline has no layers')
            return
        end if
        if (.not. c_associated(self%cuda_weights(self%token_embedding)%handle)) then
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA token embedding is not device resident')
            return
        end if
        do i = 1, size(self%layers)
            if (self%layers(i)%recurrent) then
                if (.not. c_associated(self%layers(i)%cuda_recurrent%handle)) then
                    call stat%set(FORTAI_UNSUPPORTED, 'CUDA recurrent layer is not device resident')
                    return
                end if
            else
                if (.not. c_associated(self%layers(i)%cuda_attention%handle)) then
                    call stat%set(FORTAI_UNSUPPORTED, 'CUDA attention layer is not device resident')
                    return
                end if
            end if
            if (.not. cuda_ffn_ready(self, self%layers(i))) then
                call stat%set(FORTAI_UNSUPPORTED, 'CUDA FFN layer is not device resident')
                return
            end if
            if (size(self%file%tensors(self%layers(i)%attn_norm)%bytes) /= &
                size(self%x) * storage_size(self%x(1)) / 8) then
                call stat%set(FORTAI_UNSUPPORTED, 'CUDA RMS norm weights are not F32')
                return
            end if
            if (size(self%file%tensors(self%layers(i)%post_norm)%bytes) /= &
                size(self%x) * storage_size(self%x(1)) / 8) then
                call stat%set(FORTAI_UNSUPPORTED, 'CUDA post norm weights are not F32')
                return
            end if
        end do
        if (size(self%file%tensors(self%output_norm)%bytes) /= &
            size(self%x) * storage_size(self%x(1)) / 8) then
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA output norm weights are not F32')
            return
        end if
        allocate (self%cuda_attn_norm(size(self%layers)), self%cuda_post_norm(size(self%layers)))
        do i = 1, size(self%layers)
            self%cuda_attn_norm(i) = c_null_ptr
            self%cuda_post_norm(i) = c_null_ptr
        end do
        hidden_bytes = int(size(self%x) * storage_size(self%x(1)) / 8, c_size_t)
        call self%cuda%allocate_buffer(hidden_bytes, self%cuda_x, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(hidden_bytes, self%cuda_residual, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(hidden_bytes, self%cuda_normalized, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(hidden_bytes, self%cuda_hidden, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(hidden_bytes, self%cuda_output_norm, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%upload(self%cuda_output_norm, self%file%tensors(self%output_norm)%bytes, &
            hidden_bytes, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(int(size(self%logits) * storage_size(self%logits(1)) / 8, c_size_t), &
            self%cuda_logits, stat)
        if (.not. stat%is_ok()) return
        do i = 1, size(self%layers)
            call self%cuda%allocate_buffer(hidden_bytes, self%cuda_attn_norm(i), stat)
            if (.not. stat%is_ok()) return
            call self%cuda%upload(self%cuda_attn_norm(i), &
                self%file%tensors(self%layers(i)%attn_norm)%bytes, hidden_bytes, stat)
            if (.not. stat%is_ok()) return
            call self%cuda%allocate_buffer(hidden_bytes, self%cuda_post_norm(i), stat)
            if (.not. stat%is_ok()) return
            call self%cuda%upload(self%cuda_post_norm(i), &
                self%file%tensors(self%layers(i)%post_norm)%bytes, hidden_bytes, stat)
            if (.not. stat%is_ok()) return
        end do
        self%cuda_device_pipeline = .true.
    end subroutine setup_cuda_device_pipeline

    subroutine cuda_device_pipeline_cleanup(self, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(status_t), intent(out) :: stat
        integer :: i

        call stat%clear()
        self%cuda_device_pipeline = .false.
        self%cuda_graph_ready = .false.
        if (allocated(self%cuda_attn_norm)) then
            do i = 1, size(self%cuda_attn_norm)
                if (.not. c_associated(self%cuda_attn_norm(i))) cycle
                call self%cuda%free_buffer(self%cuda_attn_norm(i), stat)
            end do
            deallocate (self%cuda_attn_norm)
        end if
        if (allocated(self%cuda_post_norm)) then
            do i = 1, size(self%cuda_post_norm)
                if (.not. c_associated(self%cuda_post_norm(i))) cycle
                call self%cuda%free_buffer(self%cuda_post_norm(i), stat)
            end do
            deallocate (self%cuda_post_norm)
        end if
        if (c_associated(self%cuda_x)) call self%cuda%free_buffer(self%cuda_x, stat)
        if (c_associated(self%cuda_residual)) call self%cuda%free_buffer(self%cuda_residual, stat)
        if (c_associated(self%cuda_normalized)) call self%cuda%free_buffer(self%cuda_normalized, stat)
        if (c_associated(self%cuda_hidden)) call self%cuda%free_buffer(self%cuda_hidden, stat)
        if (c_associated(self%cuda_output_norm)) call self%cuda%free_buffer(self%cuda_output_norm, stat)
        if (c_associated(self%cuda_logits)) call self%cuda%free_buffer(self%cuda_logits, stat)
    end subroutine cuda_device_pipeline_cleanup

    subroutine qwen35_cpu_close(self)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer :: i
        type(status_t) :: cuda_stat

        call cuda_device_pipeline_cleanup(self, cuda_stat)
        if (allocated(self%cuda_weights)) then
            if (allocated(self%layers)) then
                do i = 1, size(self%layers)
                    call self%layers(i)%cuda_recurrent%destroy(cuda_stat)
                    call self%layers(i)%cuda_attention%destroy(cuda_stat)
                end do
            end if
            do i = 1, size(self%cuda_weights)
                call self%cuda_weights(i)%destroy(cuda_stat)
            end do
            deallocate(self%cuda_weights)
        end if
        if (allocated(self%cuda_q4_weights)) then
            do i = 1, size(self%cuda_q4_weights)
                call self%cuda_q4_weights(i)%destroy(cuda_stat)
            end do
            deallocate(self%cuda_q4_weights)
        end if
        call self%cuda_q4%destroy(cuda_stat)
        call self%cuda%destroy(cuda_stat)
        self%cuda_enabled = .false.
        call gguf_quant_cache_clear()

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
        if (allocated(self % qkv_download_work)) deallocate (self % qkv_download_work)
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
        self%persistent_openmp = .false.
        self%persistent_openmp_active = .false.
    end subroutine qwen35_cpu_close

    subroutine qwen35_cpu_reset(self)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer :: i
        type(status_t) :: cuda_stat

        if (allocated(self % x)) self % x = 0.0_real32
        if (allocated(self % layers)) then
            do i = 1, size(self % layers)
                if (allocated(self % layers(i) % conv_state)) self % layers(i) % conv_state = 0.0_real32
                if (allocated(self % layers(i) % gdn_state)) self % layers(i) % gdn_state = 0.0_real32
                if (allocated(self % layers(i) % key_cache)) self % layers(i) % key_cache = 0.0_real32
                if (allocated(self%layers(i)%value_cache)) self%layers(i)%value_cache = 0.0_real32
                if (self%cuda_enabled) call self%layers(i)%cuda_recurrent%reset(cuda_stat)
                if (self%cuda_enabled) call self%layers(i)%cuda_attention%reset(cuda_stat)
            end do
        end if
    end subroutine qwen35_cpu_reset

    subroutine qwen35_cpu_forward(self, token_id, position, logits, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id, position
        real(real32), contiguous, intent(out) :: logits(:)
        type(status_t), intent(out) :: stat

        if (self%persistent_openmp .and. .not. self%cuda_device_pipeline) then
            self%persistent_openmp_active = .true.
            !$omp parallel default(none) shared(self, token_id, position, logits, stat)
            !$omp single
            call qwen35_cpu_forward_body(self, token_id, position, logits, stat)
            !$omp end single
            !$omp end parallel
            self%persistent_openmp_active = .false.
        else
            call qwen35_cpu_forward_body(self, token_id, position, logits, stat)
        end if
    end subroutine qwen35_cpu_forward

    subroutine qwen35_cpu_forward_body(self, token_id, position, logits, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id, position
        real(real32), contiguous, intent(out) :: logits(:)
        type(status_t), intent(out) :: stat
        integer :: i
        logical :: capture_graph

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
        if (self%cuda_device_pipeline) then
            call cuda_qwen35_embedding_device(self%cuda, self%cuda_weights(self%token_embedding), &
                int(token_id, c_int64_t), self%cuda_x, int(size(self%x), c_size_t), stat)
            if (.not. stat%is_ok()) return
            call self%cuda%set_position(int(position), stat)
            if (.not. stat%is_ok()) return
        else
            call self%file%tensors(self%token_embedding)%get_row(token_id + 1_int64, self%x, stat)
            if (.not. stat % is_ok()) return
        end if
        capture_graph = .false.
        if (self%cuda_device_pipeline .and. self%cuda_graph_enabled) then
            if (self%cuda_graph_ready) then
                call self%cuda%graph_launch(stat)
                if (.not. stat%is_ok()) return
            else if (position > 0_int64) then
                call self%cuda%capture_begin(stat)
                if (.not. stat%is_ok()) return
                capture_graph = .true.
            end if
        end if

        if (.not. self%cuda_device_pipeline .or. .not. self%cuda_graph_ready .or. &
            .not. self%cuda_graph_enabled) then
            do i = 1, self % layer_count
                if (self%cuda_device_pipeline) then
                    if (self%layers(i)%recurrent) then
                        call forward_recurrent_device(self, i, stat)
                        if (.not. stat%is_ok()) return
                        cycle
                    end if
                    call forward_attention_device(self, i, position, stat)
                    if (.not. stat%is_ok()) return
                    cycle
                end if
                self % residual = self % x
                call rms_norm(self % x, self % file % tensors(self % layers(i) % attn_norm), &
                    self % norm_epsilon, self % normalized, stat)
                if (.not. stat % is_ok()) return
                if (self % layers(i) % recurrent) then
                    call forward_recurrent(self, self % layers(i), i, self % normalized, stat)
                else
                    call forward_attention(self, self % layers(i), position, stat)
                end if
                if (.not. stat % is_ok()) return
                self % x = self % hidden_work + self % residual
                self % residual = self % x
                call rms_norm(self % x, self % file % tensors(self % layers(i) % post_norm), &
                    self % norm_epsilon, self % normalized, stat)
                if (.not. stat % is_ok()) return
                if (cuda_ffn_ready(self, self%layers(i))) then
                    if (mod(size(self%normalized), 32) /= 0) then
                        call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA FFN input is not block aligned')
                        return
                    end if
                    call fortai_q8_quantize(self%normalized, self%quantized_input, self%quantized_scales, &
                        int(size(self%normalized), c_int64_t))
                    call cuda_model_ffn_quantized(self, self%layers(i), size(self%normalized), &
                        self%hidden_work, stat)
                else
                    call model_matvec_pair(self, self%layers(i)%ffn_gate, self%layers(i)%ffn_up, &
                        self%normalized, self%ffn_gate_work, self%ffn_up_work, stat)
                    if (.not. stat % is_ok()) return
                    call silu_product(self % ffn_gate_work, self % ffn_up_work)
                    call model_matvec(self, self % layers(i) % ffn_down, self % ffn_gate_work, &
                        self % hidden_work, stat)
                end if
                if (.not. stat % is_ok()) return
                self % x = self % hidden_work + self % residual
            end do
        end if

        if (self%cuda_device_pipeline .and. (.not. self%cuda_graph_ready .or. &
            .not. self%cuda_graph_enabled)) then
            call forward_output_device(self, stat)
            if (.not. stat%is_ok()) return
        end if

        if (capture_graph) then
            call self%cuda%capture_end(stat)
            if (.not. stat%is_ok()) return
            self%cuda_graph_ready = .true.
            call self%cuda%graph_launch(stat)
            if (.not. stat%is_ok()) return
        end if

        if (self%cuda_device_pipeline) then
            call self%cuda%download_real(self%cuda_logits, logits, stat)
            if (.not. stat%is_ok()) return
            return
        end if
        call rms_norm(self%x, self%file%tensors(self%output_norm), self%norm_epsilon, &
            self % normalized, stat)
        if (.not. stat % is_ok()) return
        call model_matvec(self, self % output, self % normalized, logits, stat)
    end subroutine qwen35_cpu_forward_body

    subroutine enable_persistent_openmp(self)
        class(qwen35_cpu_model_t), intent(inout) :: self
        character(len=8) :: enabled
        integer :: length

        call get_environment_variable('FORTAI_ENABLE_PERSISTENT_OPENMP', enabled, length=length)
        if (length > 0) then
            if (enabled(1:length) == '1') self%persistent_openmp = .true.
        end if
    end subroutine enable_persistent_openmp

    subroutine forward_recurrent_device(self, layer_index, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: layer_index
        type(status_t), intent(out) :: stat
        integer(c_size_t) :: hidden_elements, hidden_bytes

        call stat%clear()
        hidden_elements = int(size(self%x), c_size_t)
        hidden_bytes = hidden_elements * int(storage_size(self%x(1)) / 8, c_size_t)
        call cuda_qwen35_copy_device(self%cuda, self%cuda_x, self%cuda_residual, hidden_bytes, stat)
        if (.not. stat%is_ok()) return
        call cuda_qwen35_rms_norm_device(self%cuda, self%cuda_x, &
            self%cuda_attn_norm(layer_index), self%cuda_normalized, hidden_elements, &
            self%norm_epsilon, stat)
        if (.not. stat%is_ok()) return
        call self%layers(layer_index)%cuda_recurrent%run_device(self%cuda_normalized, hidden_elements, &
            self%cuda_hidden, hidden_elements, stat)
        if (.not. stat%is_ok()) return
        call cuda_qwen35_add_device(self%cuda, self%cuda_hidden, self%cuda_residual, self%cuda_x, &
            hidden_elements, stat)
        if (.not. stat%is_ok()) return
        call cuda_qwen35_copy_device(self%cuda, self%cuda_x, self%cuda_residual, hidden_bytes, stat)
        if (.not. stat%is_ok()) return
        call cuda_qwen35_rms_norm_device(self%cuda, self%cuda_x, &
            self%cuda_post_norm(layer_index), self%cuda_normalized, hidden_elements, &
            self%norm_epsilon, stat)
        if (.not. stat%is_ok()) return
        call cuda_q8_ffn_device(self%cuda, self%cuda_weights(self%layers(layer_index)%ffn_gate), &
            self%cuda_weights(self%layers(layer_index)%ffn_up), &
            self%cuda_weights(self%layers(layer_index)%ffn_down), self%cuda_normalized, &
            hidden_elements, self%cuda_hidden, hidden_elements, stat)
        if (.not. stat%is_ok()) return
        call cuda_qwen35_add_device(self%cuda, self%cuda_hidden, self%cuda_residual, self%cuda_x, &
            hidden_elements, stat)
    end subroutine forward_recurrent_device

    subroutine forward_output_device(self, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(status_t), intent(out) :: stat
        integer(c_size_t) :: hidden_elements, logits_elements

        call stat%clear()
        hidden_elements = int(size(self%x), c_size_t)
        logits_elements = int(size(self%logits), c_size_t)
        call cuda_qwen35_rms_norm_device(self%cuda, self%cuda_x, self%cuda_output_norm, &
            self%cuda_normalized, hidden_elements, self%norm_epsilon, stat)
        if (.not. stat%is_ok()) return
        call cuda_q8_matvec_device_f32(self%cuda, self%cuda_weights(self%output), &
            self%cuda_normalized, hidden_elements, self%cuda_logits, logits_elements, stat)
    end subroutine forward_output_device

    subroutine forward_attention_device(self, layer_index, position, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: layer_index
        integer(int64), intent(in) :: position
        type(status_t), intent(out) :: stat
        integer(c_size_t) :: hidden_elements, hidden_bytes

        call stat%clear()
        hidden_elements = int(size(self%x), c_size_t)
        hidden_bytes = hidden_elements * int(storage_size(self%x(1)) / 8, c_size_t)
        call cuda_qwen35_copy_device(self%cuda, self%cuda_x, self%cuda_residual, hidden_bytes, stat)
        if (.not. stat%is_ok()) return
        call cuda_qwen35_rms_norm_device(self%cuda, self%cuda_x, &
            self%cuda_attn_norm(layer_index), self%cuda_normalized, hidden_elements, &
            self%norm_epsilon, stat)
        if (.not. stat%is_ok()) return
        call self%layers(layer_index)%cuda_attention%run_device(self%cuda_normalized, hidden_elements, &
            int(position), self%cuda_hidden, hidden_elements, stat)
        if (.not. stat%is_ok()) return
        call cuda_qwen35_add_device(self%cuda, self%cuda_hidden, self%cuda_residual, self%cuda_x, &
            hidden_elements, stat)
        if (.not. stat%is_ok()) return
        call cuda_qwen35_copy_device(self%cuda, self%cuda_x, self%cuda_residual, hidden_bytes, stat)
        if (.not. stat%is_ok()) return
        call cuda_qwen35_rms_norm_device(self%cuda, self%cuda_x, &
            self%cuda_post_norm(layer_index), self%cuda_normalized, hidden_elements, &
            self%norm_epsilon, stat)
        if (.not. stat%is_ok()) return
        call cuda_q8_ffn_device(self%cuda, self%cuda_weights(self%layers(layer_index)%ffn_gate), &
            self%cuda_weights(self%layers(layer_index)%ffn_up), &
            self%cuda_weights(self%layers(layer_index)%ffn_down), self%cuda_normalized, &
            hidden_elements, self%cuda_hidden, hidden_elements, stat)
        if (.not. stat%is_ok()) return
        call cuda_qwen35_add_device(self%cuda, self%cuda_hidden, self%cuda_residual, self%cuda_x, &
            hidden_elements, stat)
    end subroutine forward_attention_device

    subroutine forward_recurrent(self, layer, layer_index, input, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(qwen35_cpu_layer_t), intent(inout) :: layer
        integer, intent(in) :: layer_index
        real(real32), contiguous, intent(in) :: input(:)
        type(status_t), intent(out) :: stat
        integer :: channel, head, j, key_head, slot
        integer :: key_offset, query_offset, state_offset, value_offset
        integer(int64) :: tensor_index
        real(real32) :: accumulator, beta, decay, decay_factor, inverse_norm
        real(c_float) :: elapsed_ms

        call stat % clear()
        if (self%cuda_enabled .and. c_associated(self%layers(layer_index)%cuda_recurrent%handle)) then
            if (mod(size(input), 32) /= 0) then
                call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA recurrent input is not block aligned')
                return
            end if
            call fortai_q8_quantize(input, self%quantized_input, self%quantized_scales, &
                int(size(input), c_int64_t))
            call self%layers(layer_index)%cuda_recurrent%run(self%quantized_input, &
                int(size(input) + 2 * (size(input) / 32), c_size_t), self%hidden_work, &
                int(size(self%hidden_work) * storage_size(self%hidden_work(1)) / 8, c_size_t), &
                elapsed_ms, stat)
            return
        end if
        if (cuda_recurrent_projection_ready(self, layer)) then
            if (mod(size(input), 32) /= 0) then
                call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA Q8 input is not block aligned')
                return
            end if
            call fortai_q8_quantize(input, self%quantized_input, self%quantized_scales, &
                int(size(input), c_int64_t))
            call cuda_model_matvec_pair_quantized(self, layer%attn_qkv, layer%attn_gate, size(input), &
                self%qkv_work(1:self%recurrent_conv_size), self%gate_work(1:self%recurrent_inner_size), stat)
            if (.not. stat%is_ok()) return
            call cuda_model_matvec_pair_quantized(self, layer%ssm_alpha, layer%ssm_beta, size(input), &
                self%alpha_work, self%beta_work, stat)
            if (.not. stat%is_ok()) return
        else
            call model_matvec_pair(self, layer%attn_qkv, layer%attn_gate, input, &
                self%qkv_work(1:self%recurrent_conv_size), self%gate_work(1:self%recurrent_inner_size), stat)
            if (.not. stat % is_ok()) return
            call model_matvec_pair(self, layer%ssm_alpha, layer%ssm_beta, input, &
                self%alpha_work, self%beta_work, stat)
            if (.not. stat % is_ok()) return
        end if
        call fortai_silu(self % gate_work, int(self % recurrent_inner_size, c_int64_t))

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
            ! Match llama.cpp's broadcast of recurrent Q/K heads over value heads.
            key_head = mod(head - 1, self % recurrent_key_heads)
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
            inverse_norm = rms_inverse_scale(self % attention_work, &
                (head - 1) * self % recurrent_head_size + 1, self % recurrent_head_size, &
                self % norm_epsilon)
            do j = 1, self % recurrent_head_size
                self % attention_work((head - 1) * self % recurrent_head_size + j) = &
                    self % attention_work((head - 1) * self % recurrent_head_size + j) * inverse_norm * &
                    self % file % tensors(layer % ssm_norm) % value(int(j, int64)) * &
                    self % gate_work((head - 1) * self % recurrent_head_size + j)
            end do
        end do
        call model_matvec(self, layer%ssm_out, self%attention_work, self%hidden_work, stat)
    end subroutine forward_recurrent

    logical function is_q4_xl_type(value_type)
        integer(int32), intent(in) :: value_type

        is_q4_xl_type = value_type == GGML_TYPE_Q3_K .or. value_type == GGML_TYPE_Q4_K .or. &
            value_type == GGML_TYPE_Q5_K .or. value_type == GGML_TYPE_Q6_K .or. &
            value_type == GGML_TYPE_IQ4_NL .or. value_type == GGML_TYPE_IQ3_S .or. &
            value_type == GGML_TYPE_IQ4_XS
    end function is_q4_xl_type

    logical function is_unused_q4_tensor(self, tensor_index)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: tensor_index
        character(len=:), allocatable :: tensor_name

        is_unused_q4_tensor = .false.
        if (self%layer_count > 64_int32) return
        tensor_name = trim(self%file%tensors(tensor_index)%name)
        ! Some UD-Q4_K_XL files carry a spare block after the active model
        ! layers.  llama.cpp does not schedule it; avoid allocating it on
        ! either GPU while retaining all active tensors for <=64-layer models.
        if (len(tensor_name) >= 7) is_unused_q4_tensor = tensor_name(1:7) == 'blk.64.'
    end function is_unused_q4_tensor

    logical function all_q8_recurrent_weights(self, layer_index)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: layer_index
        integer :: indices(5), i

        all_q8_recurrent_weights = .false.
        indices = [self%layers(layer_index)%attn_qkv, self%layers(layer_index)%attn_gate, &
            self%layers(layer_index)%ssm_alpha, self%layers(layer_index)%ssm_beta, &
            self%layers(layer_index)%ssm_out]
        do i = 1, size(indices)
            if (indices(i) <= 0 .or. self%file%tensors(indices(i))%value_type /= GGML_TYPE_Q8_0) return
        end do
        all_q8_recurrent_weights = .true.
    end function all_q8_recurrent_weights

    logical function all_q8_attention_weights(self, layer_index)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: layer_index
        integer :: indices(4), i

        all_q8_attention_weights = .false.
        indices = [self%layers(layer_index)%attn_q, self%layers(layer_index)%attn_k, &
            self%layers(layer_index)%attn_v, self%layers(layer_index)%attn_out]
        do i = 1, size(indices)
            if (indices(i) <= 0 .or. self%file%tensors(indices(i))%value_type /= GGML_TYPE_Q8_0) return
        end do
        all_q8_attention_weights = .true.
    end function all_q8_attention_weights

    logical function cuda_recurrent_projection_ready(self, layer)
        class(qwen35_cpu_model_t), intent(in) :: self
        type(qwen35_cpu_layer_t), intent(in) :: layer

        cuda_recurrent_projection_ready = .false.
        if (.not. self%cuda_enabled) return
        if (.not. allocated(self%file%tensors)) return
        if (layer%attn_qkv <= 0 .or. layer%attn_gate <= 0 .or. layer%ssm_alpha <= 0 .or. &
            layer%ssm_beta <= 0) return
        if (self%file%tensors(layer%attn_qkv)%value_type /= GGML_TYPE_Q8_0) return
        if (self%file%tensors(layer%attn_gate)%value_type /= GGML_TYPE_Q8_0) return
        if (self%file%tensors(layer%ssm_alpha)%value_type /= GGML_TYPE_Q8_0) return
        if (self%file%tensors(layer%ssm_beta)%value_type /= GGML_TYPE_Q8_0) return
        if (.not. allocated(self%file%tensors(layer%attn_qkv)%shape)) return
        if (.not. allocated(self%file%tensors(layer%attn_gate)%shape)) return
        if (.not. allocated(self%file%tensors(layer%ssm_alpha)%shape)) return
        if (.not. allocated(self%file%tensors(layer%ssm_beta)%shape)) return
        if (size(self%file%tensors(layer%attn_qkv)%shape) /= 2) return
        if (size(self%file%tensors(layer%attn_gate)%shape) /= 2) return
        if (size(self%file%tensors(layer%ssm_alpha)%shape) /= 2) return
        if (size(self%file%tensors(layer%ssm_beta)%shape) /= 2) return
        cuda_recurrent_projection_ready = .true.
    end function cuda_recurrent_projection_ready

    logical function cuda_ffn_ready(self, layer)
        class(qwen35_cpu_model_t), intent(in) :: self
        type(qwen35_cpu_layer_t), intent(in) :: layer
        integer :: tensor_index

        cuda_ffn_ready = .false.
        if (.not. self%cuda_enabled) return
        if (.not. allocated(self%cuda_weights)) return
        if (.not. allocated(self%file%tensors)) return
        if (layer%ffn_gate <= 0 .or. layer%ffn_up <= 0 .or. layer%ffn_down <= 0) return
        if (layer%ffn_gate > size(self%cuda_weights) .or. layer%ffn_up > size(self%cuda_weights) .or. &
            layer%ffn_down > size(self%cuda_weights)) return
        if (self%file%tensors(layer%ffn_gate)%value_type /= GGML_TYPE_Q8_0 .or. &
            self%file%tensors(layer%ffn_up)%value_type /= GGML_TYPE_Q8_0 .or. &
            self%file%tensors(layer%ffn_down)%value_type /= GGML_TYPE_Q8_0) return
        if (.not. allocated(self%file%tensors(layer%ffn_gate)%shape)) return
        if (.not. allocated(self%file%tensors(layer%ffn_up)%shape)) return
        if (.not. allocated(self%file%tensors(layer%ffn_down)%shape)) return
        if (size(self%file%tensors(layer%ffn_gate)%shape) /= 2) return
        if (size(self%file%tensors(layer%ffn_up)%shape) /= 2) return
        if (size(self%file%tensors(layer%ffn_down)%shape) /= 2) return
        do tensor_index = 1, 3
            select case (tensor_index)
            case (1)
                if (.not. c_associated(self%cuda_weights(layer%ffn_gate)%handle)) return
            case (2)
                if (.not. c_associated(self%cuda_weights(layer%ffn_up)%handle)) return
            case (3)
                if (.not. c_associated(self%cuda_weights(layer%ffn_down)%handle)) return
            end select
        end do
        cuda_ffn_ready = .true.
    end function cuda_ffn_ready

    subroutine forward_attention(self, layer, position, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(qwen35_cpu_layer_t), intent(inout) :: layer
        integer(int64), intent(in) :: position
        type(status_t), intent(out) :: stat
        integer :: head, i, kv_head, q_offset, k_offset, v_offset
        real(real32) :: attention_scale

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
        attention_scale = 1.0_real32 / sqrt(real(self % attention_head_size, real32))

        do head = 1, self % attention_heads
            kv_head = (head - 1) / (self % attention_heads / self % attention_heads_kv)
            q_offset = (head - 1) * 2 * self % attention_head_size
            v_offset = (head - 1) * self % value_length
            call fortai_flash_attention_f16(self % q_work(q_offset + 1:), &
                layer % key_cache(kv_head * self % attention_head_size + 1:), &
                layer % value_cache(kv_head * self % value_length + 1:), position + 1_int64, &
                int(self % attention_head_size * self % attention_heads_kv, c_int64_t), &
                int(self % value_length * self % attention_heads_kv, c_int64_t), &
                int(self % attention_head_size, c_int64_t), int(self % value_length, c_int64_t), &
                attention_scale, self % attention_work(v_offset + 1:))
            do i = 1, self % value_length
                self % attention_work(v_offset + i) = self % attention_work(v_offset + i) * &
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
            call check_tensor_shape(self, layer % attn_norm, 1, self % hidden_size, 0, stat)
            if (.not. stat % is_ok()) return
            call check_tensor_shape(self, layer % post_norm, 1, self % hidden_size, 0, stat)
            if (.not. stat % is_ok()) return
            call check_tensor_shape(self, layer % ffn_gate, 2, self % hidden_size, &
                self % feed_forward_size, stat)
            if (.not. stat % is_ok()) return
            call check_tensor_shape(self, layer % ffn_up, 2, self % hidden_size, &
                self % feed_forward_size, stat)
            if (.not. stat % is_ok()) return
            call check_tensor_shape(self, layer % ffn_down, 2, self % feed_forward_size, &
                self % hidden_size, stat)
            if (.not. stat % is_ok()) return
            if (layer % recurrent) then
                if (layer % attn_qkv == 0 .or. layer % attn_gate == 0 .or. layer % ssm_a == 0 &
                    .or. layer % ssm_alpha == 0 .or. layer % ssm_beta == 0 .or. layer % ssm_conv == 0 &
                    .or. layer % ssm_dt == 0 .or. layer % ssm_norm == 0 .or. layer % ssm_out == 0) then
                    call stat % set(FORTAI_INVALID, 'Qwen3.5 recurrent layer is incomplete')
                else
                    call check_tensor_shape(self, layer % attn_qkv, 2, self % hidden_size, &
                        self % recurrent_conv_size, stat)
                    if (.not. stat % is_ok()) return
                    call check_tensor_shape(self, layer % attn_gate, 2, self % hidden_size, &
                        self % recurrent_inner_size, stat)
                    if (.not. stat % is_ok()) return
                    call check_tensor_shape(self, layer % ssm_a, 1, self % recurrent_value_heads, 0, stat)
                    if (.not. stat % is_ok()) return
                    call check_tensor_shape(self, layer % ssm_alpha, 2, self % hidden_size, &
                        self % recurrent_value_heads, stat)
                    if (.not. stat % is_ok()) return
                    call check_tensor_shape(self, layer % ssm_beta, 2, self % hidden_size, &
                        self % recurrent_value_heads, stat)
                    if (.not. stat % is_ok()) return
                    call check_tensor_shape(self, layer % ssm_conv, 2, self % recurrent_conv_kernel, &
                        self % recurrent_conv_size, stat)
                    if (.not. stat % is_ok()) return
                    call check_tensor_shape(self, layer % ssm_dt, 1, self % recurrent_value_heads, 0, stat)
                    if (.not. stat % is_ok()) return
                    call check_tensor_shape(self, layer % ssm_norm, 1, self % recurrent_head_size, 0, stat)
                    if (.not. stat % is_ok()) return
                    call check_tensor_shape(self, layer % ssm_out, 2, self % recurrent_inner_size, &
                        self % hidden_size, stat)
                end if
            else if (layer % attn_q == 0 .or. layer % attn_k == 0 .or. layer % attn_v == 0 &
                    .or. layer % attn_out == 0 .or. layer % q_norm == 0 .or. layer % k_norm == 0) then
                call stat % set(FORTAI_INVALID, 'Qwen3.5 attention layer is incomplete')
            else
                call check_tensor_shape(self, layer % attn_q, 2, self % hidden_size, &
                    2 * self % attention_heads * self % attention_head_size, stat)
                if (.not. stat % is_ok()) return
                call check_tensor_shape(self, layer % attn_k, 2, self % hidden_size, &
                    self % attention_heads_kv * self % attention_head_size, stat)
                if (.not. stat % is_ok()) return
                call check_tensor_shape(self, layer % attn_v, 2, self % hidden_size, &
                    self % attention_heads_kv * self % value_length, stat)
                if (.not. stat % is_ok()) return
                call check_tensor_shape(self, layer % attn_out, 2, &
                    self % attention_heads * self % value_length, self % hidden_size, stat)
                if (.not. stat % is_ok()) return
                call check_tensor_shape(self, layer % q_norm, 1, self % attention_head_size, 0, stat)
                if (.not. stat % is_ok()) return
                call check_tensor_shape(self, layer % k_norm, 1, self % attention_head_size, 0, stat)
            end if
        end associate
    end subroutine bind_layer

    subroutine check_tensor_shape(self, tensor_index, rank, first, second, stat)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: tensor_index, rank
        integer(int32), intent(in) :: first, second
        type(status_t), intent(out) :: stat

        call stat % clear()
        if (tensor_index <= 0 .or. tensor_index > size(self % file % tensors)) then
            call stat % set(FORTAI_INVALID, 'Qwen3.5 tensor binding is invalid')
            return
        end if
        if (.not. allocated(self % file % tensors(tensor_index) % shape) .or. &
            size(self % file % tensors(tensor_index) % shape) /= rank .or. &
            self % file % tensors(tensor_index) % shape(1) /= int(first, int64)) then
            call stat % set(FORTAI_INVALID, 'Qwen3.5 tensor shape does not match the model config')
            return
        end if
        if (rank == 2 .and. self % file % tensors(tensor_index) % shape(2) /= int(second, int64)) then
            call stat % set(FORTAI_INVALID, 'Qwen3.5 tensor shape does not match the model config')
        end if
    end subroutine check_tensor_shape

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

    logical function cuda_q4_single_ready(self, tensor_index)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: tensor_index

        cuda_q4_single_ready = .false.
        if (.not. self%cuda_enabled) return
        if (.not. allocated(self%file%tensors)) return
        if (tensor_index <= 0) return
        if (tensor_index > size(self%file%tensors)) return
        if (.not. is_q4_xl_type(self%file%tensors(tensor_index)%value_type)) return
        if (.not. allocated(self%cuda_q4_weights)) return
        if (tensor_index > size(self%cuda_q4_weights)) return
        if (.not. c_associated(self%cuda_q4_weights(tensor_index)%handle)) return
        cuda_q4_single_ready = .true.
    end function cuda_q4_single_ready

    logical function cpu_q4_pair_ready(self, first_index, second_index, input)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: first_index, second_index
        real(real32), intent(in) :: input(:)

        cpu_q4_pair_ready = .false.
        if (self%cuda_enabled) return
        if (.not. allocated(self%file%tensors)) return
        if (first_index <= 0 .or. second_index <= 0) return
        if (first_index > size(self%file%tensors)) return
        if (second_index > size(self%file%tensors)) return
        if (.not. is_q4_xl_type(self%file%tensors(first_index)%value_type)) return
        if (.not. is_q4_xl_type(self%file%tensors(second_index)%value_type)) return
        if (.not. allocated(self%file%tensors(first_index)%shape)) return
        if (.not. allocated(self%file%tensors(second_index)%shape)) return
        if (size(self%file%tensors(first_index)%shape) /= 2) return
        if (size(self%file%tensors(second_index)%shape) /= 2) return
        if (self%file%tensors(first_index)%shape(1) /= self%file%tensors(second_index)%shape(1)) return
        if (size(input) /= self%file%tensors(first_index)%shape(1)) return
        cpu_q4_pair_ready = .true.
    end function cpu_q4_pair_ready

    logical function cuda_q4_pair_ready(self, first_index, second_index)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: first_index, second_index

        cuda_q4_pair_ready = .false.
        if (.not. self%cuda_enabled) return
        if (.not. allocated(self%file%tensors)) return
        if (first_index <= 0 .or. second_index <= 0) return
        if (first_index > size(self%file%tensors)) return
        if (second_index > size(self%file%tensors)) return
        if (.not. is_q4_xl_type(self%file%tensors(first_index)%value_type)) return
        if (.not. is_q4_xl_type(self%file%tensors(second_index)%value_type)) return
        if (.not. allocated(self%cuda_q4_weights)) return
        if (first_index > size(self%cuda_q4_weights)) return
        if (second_index > size(self%cuda_q4_weights)) return
        if (.not. c_associated(self%cuda_q4_weights(first_index)%handle)) return
        if (.not. c_associated(self%cuda_q4_weights(second_index)%handle)) return
        cuda_q4_pair_ready = .true.
    end function cuda_q4_pair_ready

    logical function cpu_q4_triplet_ready(self, first_index, second_index, third_index, input)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: first_index, second_index, third_index
        real(real32), intent(in) :: input(:)

        cpu_q4_triplet_ready = .false.
        if (self%cuda_enabled) return
        if (.not. allocated(self%file%tensors)) return
        if (first_index <= 0 .or. second_index <= 0 .or. third_index <= 0) return
        if (first_index > size(self%file%tensors)) return
        if (second_index > size(self%file%tensors)) return
        if (third_index > size(self%file%tensors)) return
        if (.not. is_q4_xl_type(self%file%tensors(first_index)%value_type)) return
        if (.not. is_q4_xl_type(self%file%tensors(second_index)%value_type)) return
        if (.not. is_q4_xl_type(self%file%tensors(third_index)%value_type)) return
        if (.not. allocated(self%file%tensors(first_index)%shape)) return
        if (.not. allocated(self%file%tensors(second_index)%shape)) return
        if (.not. allocated(self%file%tensors(third_index)%shape)) return
        if (size(self%file%tensors(first_index)%shape) /= 2) return
        if (size(self%file%tensors(second_index)%shape) /= 2) return
        if (size(self%file%tensors(third_index)%shape) /= 2) return
        if (self%file%tensors(first_index)%shape(1) /= self%file%tensors(second_index)%shape(1)) return
        if (self%file%tensors(first_index)%shape(1) /= self%file%tensors(third_index)%shape(1)) return
        if (size(input) /= self%file%tensors(first_index)%shape(1)) return
        cpu_q4_triplet_ready = .true.
    end function cpu_q4_triplet_ready

    logical function cuda_q4_triplet_ready(self, first_index, second_index, third_index)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: first_index, second_index, third_index

        cuda_q4_triplet_ready = .false.
        if (.not. self%cuda_enabled) return
        if (.not. allocated(self%file%tensors)) return
        if (first_index <= 0 .or. second_index <= 0 .or. third_index <= 0) return
        if (first_index > size(self%file%tensors)) return
        if (second_index > size(self%file%tensors)) return
        if (third_index > size(self%file%tensors)) return
        if (.not. is_q4_xl_type(self%file%tensors(first_index)%value_type)) return
        if (.not. is_q4_xl_type(self%file%tensors(second_index)%value_type)) return
        if (.not. is_q4_xl_type(self%file%tensors(third_index)%value_type)) return
        if (.not. allocated(self%cuda_q4_weights)) return
        if (first_index > size(self%cuda_q4_weights)) return
        if (second_index > size(self%cuda_q4_weights)) return
        if (third_index > size(self%cuda_q4_weights)) return
        if (.not. c_associated(self%cuda_q4_weights(first_index)%handle)) return
        if (.not. c_associated(self%cuda_q4_weights(second_index)%handle)) return
        if (.not. c_associated(self%cuda_q4_weights(third_index)%handle)) return
        cuda_q4_triplet_ready = .true.
    end function cuda_q4_triplet_ready

    subroutine model_matvec(self, tensor_index, input, output, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: tensor_index
        real(real32), contiguous, intent(in) :: input(:)
        real(real32), contiguous, intent(out) :: output(:)
        type(status_t), intent(out) :: stat
        real(c_float) :: elapsed_ms

        call stat % clear()
        if (tensor_index == 0 .or. size(input) > size(self % quantized_input)) then
            call stat % set(FORTAI_INVALID, 'Qwen3.5 CPU matvec workspace is invalid')
            return
        end if
        if (self % file % tensors(tensor_index) % value_type == GGML_TYPE_Q8_0) then
            if (self%cuda_enabled .and. size(self%file%tensors(tensor_index)%shape) == 2) then
                if (mod(size(input), 32) /= 0) then
                    call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA Q8 input is not block aligned')
                    return
                end if
                call fortai_q8_quantize(input, self%quantized_input, self%quantized_scales, &
                    int(size(input), c_int64_t))
                call cuda_model_matvec_quantized(self, tensor_index, size(input), output, stat)
                return
            end if
            call self % file % tensors(tensor_index) % matvec_q8(input, output, &
                self % quantized_input, self % quantized_scales, stat, self%persistent_openmp_active)
        else if (cuda_q4_single_ready(self, tensor_index)) then
            call cuda_q4_matvec_host(self%cuda_q4, self%cuda_q4_weights(tensor_index), input, output, &
                elapsed_ms, stat)
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
        integer(c_int) :: code
        real(c_float) :: elapsed_ms

        call stat%clear()
        if (first_index == 0 .or. second_index == 0) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CPU paired tensor binding is invalid')
            return
        end if
        if (cpu_q4_pair_ready(self, first_index, second_index, input)) then
            code = fortai_ggml_quant_matvec_pair( &
                int(self%file%tensors(first_index)%value_type, c_int), self%file%tensors(first_index)%bytes, &
                int(size(self%file%tensors(first_index)%bytes), c_size_t), &
                self%file%tensors(first_index)%shape(2), &
                int(self%file%tensors(second_index)%value_type, c_int), self%file%tensors(second_index)%bytes, &
                int(size(self%file%tensors(second_index)%bytes), c_size_t), &
                self%file%tensors(second_index)%shape(2), self%file%tensors(first_index)%shape(1), input, &
                first_output, second_output)
            if (code == 0_c_int) return
        end if
        if (cuda_q4_pair_ready(self, first_index, second_index)) then
            call cuda_q4_matvec_host_pair(self%cuda_q4, self%cuda_q4_weights(first_index), &
                self%cuda_q4_weights(second_index), input, first_output, second_output, elapsed_ms, stat)
            return
        end if
        if (self%file%tensors(first_index)%value_type == GGML_TYPE_Q8_0 .and. &
            self%file%tensors(second_index)%value_type == GGML_TYPE_Q8_0) then
            if (self%cuda_enabled) then
                if (size(self%file%tensors(first_index)%shape) == 2) then
                    if (size(self%file%tensors(second_index)%shape) == 2) then
                        if (mod(size(input), 32) /= 0) then
                            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA Q8 input is not block aligned')
                            return
                        end if
                        call fortai_q8_quantize(input, self%quantized_input, self%quantized_scales, &
                            int(size(input), c_int64_t))
                        call cuda_model_matvec_pair_quantized(self, first_index, second_index, size(input), &
                            first_output, second_output, stat)
                        return
                    end if
                end if
            end if
            call self%file%tensors(first_index)%matvec_pair_q8( &
                self%file%tensors(second_index), input, first_output, second_output, &
                self%quantized_input, self%quantized_scales, stat, self%persistent_openmp_active)
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
        integer(c_int) :: code
        real(c_float) :: elapsed_ms

        call stat%clear()
        if (first_index == 0 .or. second_index == 0 .or. third_index == 0) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CPU triplet tensor binding is invalid')
            return
        end if
        if (cpu_q4_triplet_ready(self, first_index, second_index, third_index, input)) then
            code = fortai_ggml_quant_matvec_triplet( &
                int(self%file%tensors(first_index)%value_type, c_int), self%file%tensors(first_index)%bytes, &
                int(size(self%file%tensors(first_index)%bytes), c_size_t), &
                self%file%tensors(first_index)%shape(2), &
                int(self%file%tensors(second_index)%value_type, c_int), self%file%tensors(second_index)%bytes, &
                int(size(self%file%tensors(second_index)%bytes), c_size_t), &
                self%file%tensors(second_index)%shape(2), &
                int(self%file%tensors(third_index)%value_type, c_int), self%file%tensors(third_index)%bytes, &
                int(size(self%file%tensors(third_index)%bytes), c_size_t), &
                self%file%tensors(third_index)%shape(2), self%file%tensors(first_index)%shape(1), input, &
                first_output, second_output, third_output)
            if (code == 0_c_int) return
        end if
        if (cuda_q4_triplet_ready(self, first_index, second_index, third_index)) then
            call cuda_q4_matvec_host_triplet(self%cuda_q4, self%cuda_q4_weights(first_index), &
                self%cuda_q4_weights(second_index), self%cuda_q4_weights(third_index), input, first_output, &
                second_output, third_output, elapsed_ms, stat)
            return
        end if
        if (self%file%tensors(first_index)%value_type == GGML_TYPE_Q8_0 .and. &
            self%file%tensors(second_index)%value_type == GGML_TYPE_Q8_0 .and. &
            self%file%tensors(third_index)%value_type == GGML_TYPE_Q8_0) then
            if (self%cuda_enabled) then
                if (size(self%file%tensors(first_index)%shape) == 2) then
                    if (size(self%file%tensors(second_index)%shape) == 2) then
                        if (size(self%file%tensors(third_index)%shape) == 2) then
                            if (mod(size(input), 32) /= 0) then
                                call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA Q8 input is not block aligned')
                                return
                            end if
                            call fortai_q8_quantize(input, self%quantized_input, self%quantized_scales, &
                                int(size(input), c_int64_t))
                            call cuda_model_matvec_triplet_quantized(self, first_index, second_index, third_index, &
                                size(input), first_output, second_output, third_output, stat)
                            return
                        end if
                    end if
                end if
            end if
            call self%file%tensors(first_index)%matvec_triplet_q8( &
                self%file%tensors(second_index), self%file%tensors(third_index), input, &
                first_output, second_output, third_output, self%quantized_input, &
                self%quantized_scales, stat, self%persistent_openmp_active)
        else
            call model_matvec(self, first_index, input, first_output, stat)
            if (.not. stat%is_ok()) return
            call model_matvec(self, second_index, input, second_output, stat)
            if (.not. stat%is_ok()) return
            call model_matvec(self, third_index, input, third_output, stat)
        end if
    end subroutine model_matvec_triplet

    subroutine cuda_model_matvec_quantized(self, tensor_index, input_size, output, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: tensor_index, input_size
        real(real32), contiguous, intent(out) :: output(:)
        type(status_t), intent(out) :: stat
        integer :: block_count, activation_bytes
        real(c_float) :: elapsed_ms

        call stat%clear()
        if (.not. allocated(self%cuda_weights)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA weight binding is invalid')
            return
        end if
        if (tensor_index <= 0 .or. tensor_index > size(self%cuda_weights)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA weight binding is invalid')
            return
        end if
        if (.not. allocated(self%file%tensors(tensor_index)%shape)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA weight binding is invalid')
            return
        end if
        block_count = input_size / 32
        activation_bytes = input_size + 2 * block_count
        if (size(self%quantized_input) < activation_bytes .or. &
            size(output) /= int(self%file%tensors(tensor_index)%shape(2))) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA matvec dimensions do not agree')
            return
        end if
        call cuda_q8_matvec_host(self%cuda, self%cuda_weights(tensor_index), &
            self%quantized_input(1:activation_bytes), int(activation_bytes, c_size_t), &
            output, int(size(output) * storage_size(output(1)) / 8, c_size_t), elapsed_ms, stat)
    end subroutine cuda_model_matvec_quantized

    subroutine cuda_model_matvec_pair_quantized(self, first_index, second_index, input_size, &
            first_output, second_output, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: first_index, second_index, input_size
        real(real32), contiguous, intent(out) :: first_output(:), second_output(:)
        type(status_t), intent(out) :: stat
        integer :: block_count, activation_bytes
        real(c_float) :: elapsed_ms

        call stat%clear()
        if (.not. allocated(self%cuda_weights)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA weight binding is invalid')
            return
        end if
        if (first_index <= 0 .or. first_index > size(self%cuda_weights) .or. &
            second_index <= 0 .or. second_index > size(self%cuda_weights)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA weight binding is invalid')
            return
        end if
        if (.not. allocated(self%file%tensors)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA weight binding is invalid')
            return
        end if
        if (.not. allocated(self%file%tensors(first_index)%shape)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA weight binding is invalid')
            return
        end if
        if (.not. allocated(self%file%tensors(second_index)%shape)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA weight binding is invalid')
            return
        end if
        block_count = input_size / 32
        activation_bytes = input_size + 2 * block_count
        if (size(self%quantized_input) < activation_bytes .or. &
            size(first_output) /= int(self%file%tensors(first_index)%shape(2)) .or. &
            size(second_output) /= int(self%file%tensors(second_index)%shape(2))) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA matvec dimensions do not agree')
            return
        end if
        call cuda_q8_matvec_host_pair(self%cuda, self%cuda_weights(first_index), &
            self%cuda_weights(second_index), self%quantized_input(1:activation_bytes), &
            int(activation_bytes, c_size_t), first_output, &
            int(size(first_output) * storage_size(first_output(1)) / 8, c_size_t), second_output, &
            int(size(second_output) * storage_size(second_output(1)) / 8, c_size_t), elapsed_ms, stat)
    end subroutine cuda_model_matvec_pair_quantized

    subroutine cuda_model_matvec_triplet_quantized(self, first_index, second_index, third_index, input_size, &
            first_output, second_output, third_output, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: first_index, second_index, third_index, input_size
        real(real32), contiguous, intent(out) :: first_output(:), second_output(:), third_output(:)
        type(status_t), intent(out) :: stat
        integer :: block_count, activation_bytes
        real(c_float) :: elapsed_ms

        call stat%clear()
        if (.not. allocated(self%cuda_weights)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA weight binding is invalid')
            return
        end if
        if (first_index <= 0 .or. first_index > size(self%cuda_weights) .or. &
            second_index <= 0 .or. second_index > size(self%cuda_weights) .or. &
            third_index <= 0 .or. third_index > size(self%cuda_weights)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA weight binding is invalid')
            return
        end if
        if (.not. allocated(self%file%tensors)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA weight binding is invalid')
            return
        end if
        if (.not. allocated(self%file%tensors(first_index)%shape)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA weight binding is invalid')
            return
        end if
        if (.not. allocated(self%file%tensors(second_index)%shape)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA weight binding is invalid')
            return
        end if
        if (.not. allocated(self%file%tensors(third_index)%shape)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA weight binding is invalid')
            return
        end if
        block_count = input_size / 32
        activation_bytes = input_size + 2 * block_count
        if (size(self%quantized_input) < activation_bytes .or. &
            size(first_output) /= int(self%file%tensors(first_index)%shape(2)) .or. &
            size(second_output) /= int(self%file%tensors(second_index)%shape(2)) .or. &
            size(third_output) /= int(self%file%tensors(third_index)%shape(2))) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA matvec dimensions do not agree')
            return
        end if
        if (size(self%qkv_download_work) >= size(first_output) + size(second_output) + size(third_output)) then
            call cuda_q8_matvec_host_triplet_contiguous(self%cuda, self%cuda_weights(first_index), &
                self%cuda_weights(second_index), self%cuda_weights(third_index), &
                self%quantized_input(1:activation_bytes), int(activation_bytes, c_size_t), &
                self%qkv_download_work, &
                int((size(first_output) + size(second_output) + size(third_output)) * &
                storage_size(first_output(1)) / 8, c_size_t), elapsed_ms, stat)
            if (stat%is_ok()) then
                first_output = self%qkv_download_work(1:size(first_output))
                second_output = self%qkv_download_work(size(first_output) + 1:size(first_output) + size(second_output))
                third_output = self%qkv_download_work(size(first_output) + size(second_output) + 1: &
                    size(first_output) + size(second_output) + size(third_output))
            end if
        else
            call cuda_q8_matvec_host_triplet(self%cuda, self%cuda_weights(first_index), &
                self%cuda_weights(second_index), self%cuda_weights(third_index), &
                self%quantized_input(1:activation_bytes), int(activation_bytes, c_size_t), first_output, &
                int(size(first_output) * storage_size(first_output(1)) / 8, c_size_t), second_output, &
                int(size(second_output) * storage_size(second_output(1)) / 8, c_size_t), third_output, &
                int(size(third_output) * storage_size(third_output(1)) / 8, c_size_t), elapsed_ms, stat)
        end if
    end subroutine cuda_model_matvec_triplet_quantized

    subroutine cuda_model_ffn_quantized(self, layer, input_size, output, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(qwen35_cpu_layer_t), intent(in) :: layer
        integer, intent(in) :: input_size
        real(real32), contiguous, intent(out) :: output(:)
        type(status_t), intent(out) :: stat
        integer :: activation_bytes
        real(c_float) :: elapsed_ms

        call stat%clear()
        if (.not. allocated(self%cuda_weights) .or. .not. allocated(self%file%tensors)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA FFN weight binding is invalid')
            return
        end if
        if (layer%ffn_gate <= 0 .or. layer%ffn_up <= 0 .or. layer%ffn_down <= 0 .or. &
            layer%ffn_gate > size(self%cuda_weights) .or. layer%ffn_up > size(self%cuda_weights) .or. &
            layer%ffn_down > size(self%cuda_weights)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA FFN tensor binding is invalid')
            return
        end if
        if (mod(input_size, 32) /= 0) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA FFN width is not block aligned')
            return
        end if
        activation_bytes = input_size + 2 * (input_size / 32)
        if (size(self%quantized_input) < activation_bytes .or. size(output) /= &
            int(self%file%tensors(layer%ffn_down)%shape(2))) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA FFN dimensions do not agree')
            return
        end if
        call cuda_q8_ffn_host(self%cuda, self%cuda_weights(layer%ffn_gate), &
            self%cuda_weights(layer%ffn_up), self%cuda_weights(layer%ffn_down), &
            self%quantized_input(1:activation_bytes), int(activation_bytes, c_size_t), output, &
            int(size(output) * storage_size(output(1)) / 8, c_size_t), elapsed_ms, stat)
    end subroutine cuda_model_ffn_quantized

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
            sum_squares = sum_squares + real(input(i) * input(i), real64)
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
            sum_squares = sum_squares + real(values(i) * values(i), real64)
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
            sum_squares = sum_squares + real(values(i) * values(i), real64)
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

        sigmoid = 1.0_real32 / (1.0_real32 + exp(-value))
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

    real(real32) function rms_inverse_scale(values, first, length, epsilon)
        real(real32), intent(in) :: values(:), epsilon
        integer, intent(in) :: first, length
        integer :: i
        real(real64) :: sum_squares

        sum_squares = 0.0_real64
        do i = first, first + length - 1
            sum_squares = sum_squares + real(values(i) * values(i), real64)
        end do
        rms_inverse_scale = 1.0_real32 / &
            sqrt(real(sum_squares / real(length, real64), real32) + epsilon)
    end function rms_inverse_scale

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
