module fortai_qwen35_cpu
    use, intrinsic :: iso_c_binding, only: c_associated, c_char, c_float, c_int, c_int16_t, c_int64_t, c_int8_t, &
        c_null_char, c_null_ptr, c_ptr, c_size_t
    use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real32, real64
    use fortai_backend_cuda, only: cuda_q8_context_t, cuda_q8_matvec_host, &
        cuda_q8_ffn_host, cuda_q8_ffn_device, cuda_q8_matvec_host_pair, &
        cuda_q8_matvec_host_triplet, cuda_q8_matvec_host_triplet_contiguous, cuda_q8_weights_t, &
        cuda_q8_matvec_device_f32, cuda_qwen35_add_device, cuda_qwen35_copy_device, &
        cuda_qwen35_embedding_device, cuda_qwen35_rms_norm_device
    use fortai_backend_cuda, only: cuda_qwen35_silu_product_device
    use fortai_backend_cuda, only: cuda_q4_context_t, cuda_q4_weights_t, cuda_q4_matvec_host, &
        cuda_q4_matvec_host_pair, cuda_q4_matvec_host_triplet, cuda_q4_matvec_device, &
        cuda_q4_matvec_device_pair, cuda_q4_matvec_device_triplet, cuda_q4_embedding_device
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

        integer(c_int) function fortai_llama_fast_available() bind(C, name='fortai_llama_fast_available')
            import c_int
        end function fortai_llama_fast_available

        integer(c_int) function fortai_llama_fast_context_create(path, context_size, threads, gpu_layers, &
                main_gpu, handle, vocab, layers) bind(C, name='fortai_llama_fast_context_create')
            import c_char, c_int, c_ptr
            character(kind=c_char), intent(in) :: path(*)
            integer(c_int), value, intent(in) :: context_size, threads, gpu_layers, main_gpu
            type(c_ptr), intent(out) :: handle
            integer(c_int), intent(out) :: vocab, layers
        end function fortai_llama_fast_context_create

        integer(c_int) function fortai_llama_fast_context_decode(handle, token, position, logits, count) &
                bind(C, name='fortai_llama_fast_context_decode')
            import c_float, c_int, c_ptr, c_size_t
            type(c_ptr), value, intent(in) :: handle
            integer(c_int), value, intent(in) :: token, position
            real(c_float), intent(out) :: logits(*)
            integer(c_size_t), value, intent(in) :: count
        end function fortai_llama_fast_context_decode

        integer(c_int) function fortai_llama_fast_context_decode_greedy(handle, token, position, next_token, logit_sum) &
                bind(C, name='fortai_llama_fast_context_decode_greedy')
            import c_float, c_int, c_ptr
            type(c_ptr), value, intent(in) :: handle
            integer(c_int), value, intent(in) :: token, position
            integer(c_int), intent(out) :: next_token
            real(c_float), intent(out) :: logit_sum
        end function fortai_llama_fast_context_decode_greedy

        integer(c_int) function fortai_llama_fast_context_decode_speculative(handle, token, position, &
                tokens, capacity, count, logit_sum) bind(C, name='fortai_llama_fast_context_decode_speculative')
            import c_float, c_int, c_ptr
            type(c_ptr), value, intent(in) :: handle
            integer(c_int), value, intent(in) :: token, position, capacity
            integer(c_int), intent(out) :: tokens(*), count
            real(c_float), intent(out) :: logit_sum
        end function fortai_llama_fast_context_decode_speculative

        integer(c_int) function fortai_llama_fast_context_reset(handle) &
                bind(C, name='fortai_llama_fast_context_reset')
            import c_int, c_ptr
            type(c_ptr), value, intent(in) :: handle
        end function fortai_llama_fast_context_reset

        integer(c_int) function fortai_llama_fast_context_destroy(handle) &
                bind(C, name='fortai_llama_fast_context_destroy')
            import c_int, c_ptr
            type(c_ptr), value, intent(in) :: handle
        end function fortai_llama_fast_context_destroy
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
        integer(int8), allocatable :: key_cache_q8(:)
        integer(int8), allocatable :: value_cache_q8(:)
        real(real32), allocatable :: key_cache_q8_scales(:)
        real(real32), allocatable :: value_cache_q8_scales(:)
        type(cuda_qwen35_recurrent_t) :: cuda_recurrent
        type(cuda_qwen35_attention_t) :: cuda_attention
    end type qwen35_cpu_layer_t

    type, public :: qwen35_cpu_model_t
        type(gguf_file_t) :: file
        type(qwen35_cpu_layer_t), allocatable :: layers(:)
        type(qwen35_cpu_layer_t) :: mtp_layer
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
        integer(int32) :: mtp_eh_proj = 0_int32
        integer(int32) :: mtp_enorm = 0_int32
        integer(int32) :: mtp_hnorm = 0_int32
        integer(int32) :: mtp_embed_tokens = 0_int32
        integer(int32) :: mtp_shared_head_norm = 0_int32
        integer(int32) :: mtp_shared_head_head = 0_int32
        integer(int32) :: mtp_output = 0_int32
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
        real(real32), allocatable :: attention_key_work(:)
        real(real32), allocatable :: attention_value_work(:)
        real(real32), allocatable :: ffn_gate_work(:)
        real(real32), allocatable :: ffn_up_work(:)
        real(real32), allocatable :: conv_work(:)
        real(real32), allocatable :: beta_work(:)
        real(real32), allocatable :: alpha_work(:)
        real(real32), allocatable :: logits(:)
        real(real32), allocatable :: mtp_target_hidden(:)
        real(real32), allocatable :: mtp_pending_hidden(:)
        real(real32), allocatable :: mtp_embedding(:)
        real(real32), allocatable :: mtp_concat(:)
        real(real32), allocatable :: mtp_logits(:)
        integer(int8), allocatable :: quantized_input(:)
        real(real32), allocatable :: quantized_scales(:)
        type(cuda_q8_context_t) :: cuda
        type(cuda_q8_context_t) :: cuda_second
        type(cuda_q8_weights_t), allocatable :: cuda_weights(:)
        integer, allocatable :: cuda_weight_device(:)
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
        type(c_ptr) :: cuda_qkv_device = c_null_ptr
        type(c_ptr) :: cuda_gate_device = c_null_ptr
        type(c_ptr) :: cuda_alpha_device = c_null_ptr
        type(c_ptr) :: cuda_beta_device = c_null_ptr
        type(c_ptr) :: cuda_ffn_gate_device = c_null_ptr
        type(c_ptr) :: cuda_ffn_up_device = c_null_ptr
        type(c_ptr) :: cuda_attention_q_device = c_null_ptr
        type(c_ptr) :: cuda_attention_k_device = c_null_ptr
        type(c_ptr) :: cuda_attention_v_device = c_null_ptr
        type(c_ptr) :: cuda_attention_work_device = c_null_ptr
        type(c_ptr) :: fast_handle = c_null_ptr
        character(len=:), allocatable :: model_path
        logical :: cuda_enabled = .false.
        logical :: cuda_device_pipeline = .false.
        logical :: cuda_graph_enabled = .false.
        logical :: cuda_graph_ready = .false.
        logical :: cuda_q4_resident = .false.
        logical :: cuda_q4_split = .false.
        logical :: cuda_q8_split = .false.
        logical :: cuda_q8_cpu_override = .false.
        logical :: cuda_q4_group_enabled = .true.
        logical :: fast_enabled = .false.
        logical :: fast_gpu = .false.
        logical :: mtp_available = .false.
        logical :: mtp_active = .false.
        integer(int64) :: mtp_last_pair_position = -1_int64
        integer(int64) :: mtp_last_target_position = -1_int64
        integer(int64) :: mtp_last_pair_token = -1_int64
        integer(int64) :: mtp_last_draft_token = -1_int64
        logical :: mtp_last_draft_match = .false.
        logical :: persistent_openmp = .false.
        logical :: persistent_openmp_active = .false.
        logical :: cache_key_q8 = .false.
        logical :: cache_value_q8 = .false.
        logical :: flash_attention_enabled = .true.
        integer(int32) :: cache_type_k = 1_int32
        integer(int32) :: cache_type_v = 1_int32
    contains
        procedure :: close => qwen35_cpu_close
        procedure :: enable_cuda => qwen35_cpu_enable_cuda
        procedure :: forward => qwen35_cpu_forward
        procedure :: forward_greedy => qwen35_cpu_forward_greedy
        procedure :: forward_greedy_speculative => qwen35_cpu_forward_greedy_speculative
        procedure :: mtp_draft_greedy => qwen35_cpu_mtp_draft_greedy
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
        integer(int32) :: i, interval, raw_layer_count, mtp_block
        integer :: work_size
        logical :: mtp_schema

        call self % close()
        self%model_path = trim(path)
        self%max_context = 256_int64
        if (present(max_context)) self%max_context = max_context
        self%persistent_openmp = .false.
        if ((fast_path_mode() == 1 .or. fast_path_mode() == 3) .and. &
            .not. native_mtp_requested()) then
            call fast_context_create(self, 0, 0, stat)
            if (stat%is_ok()) then
                self%fast_gpu = .false.
                return
            end if
            call stat%clear()
            call fast_context_destroy(self)
        end if
        call self % file % open(path, stat)
        if (.not. stat % is_ok()) return

        self%hidden_size = int(self%file%meta_int('qwen35.embedding_length', 0_int64), int32)
        self%vocabulary_size = int(self%file%meta_int('qwen35.vocab_size', 0_int64), int32)
        raw_layer_count = int(self % file % meta_int('qwen35.block_count', 0_int64), int32)
        self % layer_count = raw_layer_count
        self%mtp_available = .false.
        self%mtp_active = .false.
        self%mtp_last_pair_position = -1_int64
        self%mtp_last_target_position = -1_int64
        self%mtp_last_pair_token = -1_int64
        self%mtp_last_draft_token = -1_int64
        self%mtp_last_draft_match = .false.
        ! Qwen3.8-27B Q4_K_XL carries one MTP/nextn block after the base
        ! transformer.  Keep the base block count separate and bind the
        ! optional NextN block below when the caller explicitly requests it.
        mtp_schema = .false.
        mtp_block = raw_layer_count - 1
        if (raw_layer_count > 1 .and. find_layer_tensor(self%file, mtp_block, &
            'nextn.eh_proj.weight') > 0) then
            mtp_schema = .true.
            self % layer_count = mtp_block
        end if
        if (native_mtp_requested()) then
            call validate_mtp_sidecar(stat)
            if (.not. stat%is_ok()) return
        end if
        if (native_mtp_requested() .and. .not. mtp_schema) then
            call stat%set(FORTAI_UNSUPPORTED, 'native MTP was requested but the GGUF has no NextN block')
            return
        end if
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
        call configure_kv_cache(self, stat)
        if (.not. stat%is_ok()) return
        call configure_flash_attention(self, stat)
        if (.not. stat%is_ok()) return

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
                call allocate_attention_cache(self, self%layers(i))
            end if
        end do

        if (mtp_schema) then
            call bind_mtp_layer(self, mtp_block, stat)
            if (.not. stat%is_ok()) then
                if (native_mtp_requested()) then
                    return
                end if
                call stat%clear()
                self%mtp_eh_proj = 0
            else
                self%mtp_available = .true.
                self%mtp_active = native_mtp_requested()
                call allocate_attention_cache(self, self%mtp_layer)
            end if
        end if

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
        if (self%cache_key_q8) allocate(self%attention_key_work(self%attention_head_size * self%max_context))
        if (self%cache_value_q8) allocate(self%attention_value_work(self%value_length * self%max_context))
        allocate (self % ffn_gate_work(self % feed_forward_size))
        allocate (self % ffn_up_work(self % feed_forward_size))
        allocate (self % conv_work(self % recurrent_conv_size))
        allocate (self % beta_work(self % recurrent_value_heads))
        allocate (self % alpha_work(self % recurrent_value_heads))
        allocate (self % logits(self % vocabulary_size))
        if (self%mtp_available) then
            allocate(self%mtp_target_hidden(self%hidden_size))
            allocate(self%mtp_pending_hidden(self%hidden_size))
            allocate(self%mtp_embedding(self%hidden_size))
            allocate(self%mtp_concat(2 * self%hidden_size))
            allocate(self%mtp_logits(self%vocabulary_size))
        end if
        allocate (self % quantized_input(work_size + 2 * ((work_size + 31) / 32)))
        allocate (self % quantized_scales((work_size + 31) / 32))
        call self % reset()
    end subroutine qwen35_cpu_open

    subroutine configure_kv_cache(self, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(status_t), intent(out) :: stat
        character(len=32) :: key_type, value_type
        integer :: key_length, value_length

        call stat%clear()
        key_type = 'f16'
        value_type = 'f16'
        key_length = 0
        value_length = 0
        call get_environment_variable('FORTAI_CACHE_TYPE_K', key_type, length=key_length)
        if (key_length <= 0) call get_environment_variable('LLAMACPP_CACHE_TYPE_K', key_type, length=key_length)
        call get_environment_variable('FORTAI_CACHE_TYPE_V', value_type, length=value_length)
        if (value_length <= 0) call get_environment_variable('LLAMACPP_CACHE_TYPE_V', value_type, length=value_length)
        if (key_length <= 0) then
            key_type = 'f16'
            key_length = 3
        end if
        if (value_length <= 0) then
            value_type = 'f16'
            value_length = 3
        end if
        if (key_length > len(key_type) .or. value_length > len(value_type)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 KV cache type is too long')
            return
        end if
        select case (trim(key_type(:key_length)))
        case ('f32', 'f16')
            self%cache_type_k = merge(0_int32, 1_int32, trim(key_type(:key_length)) == 'f32')
        case ('q8_0')
            self%cache_type_k = 8_int32
            self%cache_key_q8 = .true.
        case default
            call stat%set(FORTAI_UNSUPPORTED, 'native K cache supports only f32, f16, and q8_0')
            return
        end select
        select case (trim(value_type(:value_length)))
        case ('f32', 'f16')
            self%cache_type_v = merge(0_int32, 1_int32, trim(value_type(:value_length)) == 'f32')
        case ('q8_0')
            self%cache_type_v = 8_int32
            self%cache_value_q8 = .true.
        case default
            call stat%set(FORTAI_UNSUPPORTED, 'native V cache supports only f32, f16, and q8_0')
            return
        end select
        if ((self%cache_key_q8 .and. mod(self%attention_head_size, 32) /= 0) .or. &
            (self%cache_value_q8 .and. mod(self%value_length, 32) /= 0)) then
            call stat%set(FORTAI_UNSUPPORTED, 'native q8_0 KV cache requires 32-element head and value blocks')
            return
        end if
    end subroutine configure_kv_cache

    subroutine configure_flash_attention(self, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(status_t), intent(out) :: stat
        character(len=16) :: value
        integer :: length

        call stat%clear()
        self%flash_attention_enabled = .true.
        value = ''
        call get_environment_variable('FORTAI_FLASH_ATTN', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_FLASH_ATTN', value, length=length)
        if (length <= 0) return
        if (length > len(value)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 flash-attention mode is too long')
            return
        end if
        select case (trim(value(:length)))
        case ('on', 'true', '1', 'yes', 'auto')
            self%flash_attention_enabled = .true.
        case ('off', 'false', '0', 'no')
            self%flash_attention_enabled = .false.
        case default
            call stat%set(FORTAI_INVALID, 'Qwen3.5 flash-attention mode must be on, off, or auto')
        end select
    end subroutine configure_flash_attention

    subroutine allocate_attention_cache(self, layer)
        class(qwen35_cpu_model_t), intent(in) :: self
        type(qwen35_cpu_layer_t), intent(inout) :: layer
        integer :: key_blocks, value_blocks, key_row, value_row

        key_blocks = (self%attention_head_size + 31) / 32
        value_blocks = (self%value_length + 31) / 32
        key_row = self%attention_head_size + 2 * key_blocks
        value_row = self%value_length + 2 * value_blocks
        if (self%cache_key_q8) then
            if (.not. allocated(layer%key_cache_q8)) then
                allocate(layer%key_cache_q8(key_row * self%attention_heads_kv * self%max_context))
            end if
            if (.not. allocated(layer%key_cache_q8_scales)) then
                allocate(layer%key_cache_q8_scales(key_blocks * self%attention_heads_kv * self%max_context))
            end if
        else
            if (.not. allocated(layer%key_cache)) then
                allocate(layer%key_cache(self%attention_head_size * self%attention_heads_kv * self%max_context))
            end if
        end if
        if (self%cache_value_q8) then
            if (.not. allocated(layer%value_cache_q8)) then
                allocate(layer%value_cache_q8(value_row * self%attention_heads_kv * self%max_context))
            end if
            if (.not. allocated(layer%value_cache_q8_scales)) then
                allocate(layer%value_cache_q8_scales(value_blocks * self%attention_heads_kv * self%max_context))
            end if
        else
            if (.not. allocated(layer%value_cache)) then
                allocate(layer%value_cache(self%value_length * self%attention_heads_kv * self%max_context))
            end if
        end if
    end subroutine allocate_attention_cache

    logical function qwen35_cuda_second_requested(self, device, second_device)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: device
        integer, intent(out) :: second_device
        character(len=8) :: split_value
        character(len=32) :: device_value
        character(len=8) :: pipeline_value
        integer :: split_length, device_length, pipeline_length, ios, i

        qwen35_cuda_second_requested = .false.
        second_device = device + 1
        if (.not. allocated(self%file%tensors)) return
        ! The all-device pipeline keeps its Q8 activations and scratch on the
        ! primary CUDA context.  Do not place every Q8 weight on the peer GPU
        ! in that mode; the mixed Q4 bridge still distributes its own tensors,
        ! while a peer-resident Q8 output would fail the context ownership
        ! check and require an extra cross-device copy for every matvec.
        pipeline_value = ''
        call get_environment_variable('FORTAI_ENABLE_CUDA_Q4_DEVICE_PIPELINE', pipeline_value, &
            length=pipeline_length)
        if (pipeline_length > 0) then
            if (pipeline_value(1:1) == '1') return
        end if
        do i = 1, size(self%file%tensors)
            if (is_q4_xl_type(self%file%tensors(i)%value_type)) then
                qwen35_cuda_second_requested = .true.
                exit
            end if
        end do
        if (.not. qwen35_cuda_second_requested) return
        split_value = ''
        call get_environment_variable('FORTAI_CUDA_Q4_SPLIT', split_value, length=split_length)
        if (split_length > 0 .and. split_value(1:1) == '0') then
            qwen35_cuda_second_requested = .false.
            return
        end if
        device_value = ''
        call get_environment_variable('FORTAI_CUDA_Q4_SECOND_DEVICE', device_value, length=device_length)
        if (device_length > 0) then
            read(device_value(:min(device_length, len(device_value))), *, iostat=ios) second_device
            if (ios /= 0) second_device = device + 1
        end if
        if (second_device == device) qwen35_cuda_second_requested = .false.
    end function qwen35_cuda_second_requested

    subroutine qwen35_cpu_enable_cuda(self, device, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: device
        type(status_t), intent(out) :: stat
        type(status_t) :: cleanup_stat
        integer :: i, j, rows, width, q4_device, q4_second_device, q8_second_device
        integer(int64) :: q4_bytes(2)
        logical :: have_q4, q4_split
        logical :: tensor_split_custom
        real(real64) :: tensor_split_fraction(2), tensor_split_sum
        character(len=8) :: resident_env
        character(len=8) :: group_env
        character(len=128) :: tensor_split_env
        integer :: resident_length
        integer :: group_length
        integer :: tensor_split_length, tensor_split_status

        call stat%clear()
        self%cuda_q4_group_enabled = .true.
        group_env = ''
        call get_environment_variable('FORTAI_CUDA_Q4_GROUP', group_env, length=group_length)
        if (group_length > 0 .and. group_env(1:1) == '0') self%cuda_q4_group_enabled = .false.
        call fast_context_destroy(self)
        self%fast_gpu = .false.
        if ((fast_path_mode() == 2 .or. fast_path_mode() == 3) .and. &
            .not. native_mtp_requested()) then
            call fast_context_create(self, -1, device, stat)
            if (stat%is_ok()) then
                self%fast_gpu = .true.
                self%cuda_enabled = .false.
                return
            end if
            call stat%clear()
            call fast_context_destroy(self)
        end if
        if (.not. allocated(self%file%tensors)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA model is not open')
            return
        end if
        if (allocated(self%layers)) then
            do i = 1, size(self%layers)
                if (.not. self%layers(i)%recurrent) call allocate_attention_cache(self, self%layers(i))
            end do
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
            if (allocated(self%cuda_weight_device)) deallocate(self%cuda_weight_device)
        end if
        if (allocated(self%cuda_q4_weights)) then
            do i = 1, size(self%cuda_q4_weights)
                call self%cuda_q4_weights(i)%destroy(cleanup_stat)
            end do
            deallocate(self%cuda_q4_weights)
        end if
        call self%cuda_q4%destroy(cleanup_stat)
        self%cuda_enabled = .false.
        self%cuda_q4_resident = .false.
        self%cuda_q4_split = .false.
        self%cuda_q8_split = .false.
        self%cuda_q8_cpu_override = .false.
        call self%cuda_second%destroy(cleanup_stat)
        call self%cuda%destroy(cleanup_stat)
        call self%cuda%create(device, stat)
        if (.not. stat%is_ok()) return

        self%cuda_q8_split = qwen35_cuda_second_requested(self, device, q8_second_device)
        if (self%cuda_q8_split) then
            call self%cuda_second%create(q8_second_device, stat)
            if (.not. stat%is_ok()) then
                call self%cuda_second%destroy(cleanup_stat)
                call self%cuda%destroy(cleanup_stat)
                return
            end if
        end if

        allocate(self%cuda_weights(size(self%file%tensors)))
        allocate(self%cuda_weight_device(size(self%file%tensors)))
        self%cuda_weight_device = 1
        do i = 1, size(self%file%tensors)
            if (self%file%tensors(i)%value_type /= GGML_TYPE_Q8_0) cycle
            if (size(self%file%tensors(i)%shape) /= 2) cycle
            width = int(self%file%tensors(i)%shape(1))
            rows = int(self%file%tensors(i)%shape(2))
            if (self%cuda_q8_split) then
                self%cuda_weight_device(i) = 2
                call self%cuda_weights(i)%upload(self%cuda_second, self%file%tensors(i)%bytes, &
                    int(size(self%file%tensors(i)%bytes), c_size_t), rows, width, stat)
            else
                call self%cuda_weights(i)%upload(self%cuda, self%file%tensors(i)%bytes, &
                    int(size(self%file%tensors(i)%bytes), c_size_t), rows, width, stat)
            end if
            if (.not. stat%is_ok()) then
                do j = 1, i - 1
                    call self%cuda_weights(j)%destroy(cleanup_stat)
                end do
                deallocate(self%cuda_weights)
                if (allocated(self%cuda_weight_device)) deallocate(self%cuda_weight_device)
                call self%cuda_second%destroy(cleanup_stat)
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
            self%cuda_q4_resident = .true.
            q4_split = .true.
            tensor_split_custom = .false.
            tensor_split_fraction = 0.5_real64
            resident_env = ''
            call get_environment_variable('FORTAI_CUDA_Q4_SPLIT', resident_env, length=resident_length)
            if (resident_length > 0) then
                if (resident_env(1:1) == '0') q4_split = .false.
            end if
            q4_second_device = device + 1
            ! The default pair is device/device+1; an explicit override is
            ! parsed below without making the Q8 single-device path depend on it.
            block
                character(len=32) :: second_text
                integer :: second_length, second_status
                call get_environment_variable('FORTAI_CUDA_Q4_SECOND_DEVICE', second_text, length=second_length)
                if (second_length > 0) read(second_text(1:second_length), *, iostat=second_status) q4_second_device
            end block
            if (.not. q4_split) q4_second_device = device
            self%cuda_q4_split = q4_split .and. q4_second_device /= device
            if (self%cuda_q4_split) then
                tensor_split_env = ''
                call get_environment_variable('FORTAI_TENSOR_SPLIT', tensor_split_env, &
                    length=tensor_split_length)
                if (tensor_split_length > 0) then
                    tensor_split_fraction = 0.0_real64
                    tensor_split_status = 0
                    if (tensor_split_length > len(tensor_split_env)) then
                        tensor_split_status = 1
                    else
                        read(tensor_split_env(1:tensor_split_length), *, iostat=tensor_split_status) &
                            tensor_split_fraction
                    end if
                    tensor_split_sum = sum(tensor_split_fraction)
                    if (tensor_split_status /= 0 .or. any(tensor_split_fraction < 0.0_real64) .or. &
                        tensor_split_sum <= 0.0_real64) then
                        call stat%set(FORTAI_INVALID, &
                            'FORTAI_TENSOR_SPLIT must contain two non-negative fractions with a positive sum')
                        call self%cuda_q4%destroy(cleanup_stat)
                        do j = 1, size(self%cuda_weights)
                            call self%cuda_weights(j)%destroy(cleanup_stat)
                        end do
                        deallocate(self%cuda_weights)
                        if (allocated(self%cuda_weight_device)) deallocate(self%cuda_weight_device)
                        call self%cuda_second%destroy(cleanup_stat)
                        call self%cuda%destroy(cleanup_stat)
                        return
                    end if
                    tensor_split_fraction = tensor_split_fraction / tensor_split_sum
                    tensor_split_custom = .true.
                end if
            end if
            call self%cuda_q4%create(device, q4_second_device, stat)
            if (.not. stat%is_ok()) then
                do j = 1, size(self%cuda_weights)
                    call self%cuda_weights(j)%destroy(cleanup_stat)
                end do
                deallocate(self%cuda_weights)
                if (allocated(self%cuda_weight_device)) deallocate(self%cuda_weight_device)
                call self%cuda_second%destroy(cleanup_stat)
                call self%cuda%destroy(cleanup_stat)
                return
            end if
            allocate(self%cuda_q4_weights(size(self%file%tensors)))
            q4_bytes = 0_int64
            do i = 1, size(self%file%tensors)
                if (.not. is_q4_xl_type(self%file%tensors(i)%value_type)) cycle
                if (size(self%file%tensors(i)%shape) /= 2) cycle
                if (is_unused_q4_tensor(self, i)) cycle
                if (.not. q4_split) then
                    q4_device = 0
                else if (tensor_split_custom .and. tensor_split_fraction(1) <= 0.0_real64) then
                    q4_device = 1
                else if (tensor_split_custom .and. tensor_split_fraction(2) <= 0.0_real64) then
                    q4_device = 0
                else if (tensor_split_custom) then
                    if (real(q4_bytes(1), real64) / tensor_split_fraction(1) <= &
                        real(q4_bytes(2), real64) / tensor_split_fraction(2)) then
                        q4_device = 0
                    else
                        q4_device = 1
                    end if
                else
                    q4_device = 1
                    if (q4_bytes(1) <= q4_bytes(2)) q4_device = 0
                end if
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
                    if (allocated(self%cuda_weight_device)) deallocate(self%cuda_weight_device)
                    call self%cuda_second%destroy(cleanup_stat)
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
            ! GGML's Q4 scheduler runs on a separate stream.  Attach the
            ! native Q8 stream so the bridge can hand results across with a
            ! CUDA event instead of synchronizing the whole device.
            call self%cuda_q4%set_consumer_stream(0, self%cuda%stream(), stat)
            if (.not. stat%is_ok()) then
                do j = 1, size(self%cuda_q4_weights)
                    call self%cuda_q4_weights(j)%destroy(cleanup_stat)
                end do
                deallocate(self%cuda_q4_weights)
                do j = 1, size(self%cuda_weights)
                    call self%cuda_weights(j)%destroy(cleanup_stat)
                end do
                deallocate(self%cuda_weights)
                call self%cuda_q4%destroy(cleanup_stat)
                call self%cuda%destroy(cleanup_stat)
                return
            end if
        end if
        do i = 1, size(self%layers)
            if (.not. self%layers(i)%recurrent) cycle
            if (all_q8_recurrent_weights(self, i)) then
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
            else if (cuda_recurrent_device_ready(self, i)) then
                call self%layers(i)%cuda_recurrent%create_state(self%cuda, &
                    self%file%tensors(self%layers(i)%ssm_conv)%bytes, &
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
            else
                cycle
            end if
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
            if (.not. cuda_attention_device_ready(self, i)) cycle
            if (size(self%file%tensors(self%layers(i)%q_norm)%bytes) /= &
                self%attention_head_size * storage_size(self%x(1)) / 8) cycle
            if (size(self%file%tensors(self%layers(i)%k_norm)%bytes) /= &
                self%attention_head_size * storage_size(self%x(1)) / 8) cycle
            if (all_q8_attention_weights(self, i)) then
                call self%layers(i)%cuda_attention%create(self%cuda, &
                    self%cuda_weights(self%layers(i)%attn_q), self%cuda_weights(self%layers(i)%attn_k), &
                    self%cuda_weights(self%layers(i)%attn_v), self%cuda_weights(self%layers(i)%attn_out), &
                    self%file%tensors(self%layers(i)%q_norm)%bytes, &
                    int(size(self%file%tensors(self%layers(i)%q_norm)%bytes), c_size_t), &
                    self%file%tensors(self%layers(i)%k_norm)%bytes, &
                    int(size(self%file%tensors(self%layers(i)%k_norm)%bytes), c_size_t), &
                    self%attention_heads, self%attention_heads_kv, self%attention_head_size, &
                    self%value_length, int(self%max_context), self%rope_dimension, self%rope_base, &
                    self%norm_epsilon, cleanup_stat, self%cache_key_q8, self%cache_value_q8)
            else
                call self%layers(i)%cuda_attention%create_state(self%cuda, &
                    self%file%tensors(self%layers(i)%q_norm)%bytes, &
                    int(size(self%file%tensors(self%layers(i)%q_norm)%bytes), c_size_t), &
                    self%file%tensors(self%layers(i)%k_norm)%bytes, &
                    int(size(self%file%tensors(self%layers(i)%k_norm)%bytes), c_size_t), &
                    self%attention_heads, self%attention_heads_kv, self%attention_head_size, &
                    self%value_length, int(self%max_context), self%rope_dimension, self%rope_base, &
                    self%norm_epsilon, cleanup_stat, self%cache_key_q8, self%cache_value_q8)
            end if
            if (.not. cleanup_stat%is_ok()) call cleanup_stat%clear()
        end do
        self%cuda_enabled = .true.
        call setup_cuda_device_pipeline(self, cleanup_stat)
        if (.not. cleanup_stat%is_ok()) call cleanup_stat%clear()
    end subroutine qwen35_cpu_enable_cuda

    subroutine setup_cuda_device_pipeline(self, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(status_t), intent(out) :: stat
        integer :: i, graph_length, pipeline_length, q4_pipeline_length
        integer(c_size_t) :: hidden_bytes, ffn_bytes, qkv_bytes, query_bytes, key_bytes, value_bytes
        integer(c_size_t) :: core_bytes
        character(len=8) :: graph_env
        character(len=8) :: pipeline_env
        character(len=8) :: q4_pipeline_env

        call stat%clear()
        self%cuda_device_pipeline = .false.
        self%cuda_graph_enabled = .false.
        self%cuda_graph_ready = .false.
        ! Native MTP consumes the target pre-output hidden state on the host
        ! and shares the host KV implementation for its attention block.  Do
        ! not mix that stateful hand-off with the opt-in device pipeline until
        ! its longer-sequence oracle is closed.
        if (self%mtp_active) return
        ! Q8_0 K/V caches are now first-class CUDA-resident storage.  The
        ! attention ABI carries the independent K/V flags, so mixed f16/q8
        ! configurations use the same resident pipeline without a host
        ! dequantization round trip.
        if (.not. self%flash_attention_enabled) return
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
        ! Mixed-device Q4_K_XL currently uses GGML-CUDA on a second scheduler
        ! and peer bridges.  Keep the production resident path deterministic
        ! by using the verified host-boundary CUDA route unless explicitly
        ! opting into the diagnostic all-device bridge.
        if (self%cuda_q4_split) then
            q4_pipeline_env = ''
            call get_environment_variable('FORTAI_ENABLE_CUDA_Q4_DEVICE_PIPELINE', &
                q4_pipeline_env, length=q4_pipeline_length)
            if (q4_pipeline_length <= 0 .or. q4_pipeline_env(1:1) /= '1') return
        end if
        if (.not. allocated(self%layers)) then
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA device pipeline has no layers')
            return
        end if
        if (.not. cuda_quantized_device_ready(self, self%token_embedding)) then
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
            if (.not. cuda_ffn_device_ready(self, self%layers(i))) then
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
        ffn_bytes = int(self%feed_forward_size, c_size_t) * int(storage_size(self%x(1)) / 8, c_size_t)
        qkv_bytes = int(max(self%recurrent_conv_size, 2 * self%attention_heads * self%attention_head_size), &
            c_size_t) * int(storage_size(self%x(1)) / 8, c_size_t)
        query_bytes = int(2 * self%attention_heads * self%attention_head_size, c_size_t) * &
            int(storage_size(self%x(1)) / 8, c_size_t)
        key_bytes = int(self%attention_heads_kv * self%attention_head_size, c_size_t) * &
            int(storage_size(self%x(1)) / 8, c_size_t)
        value_bytes = int(self%attention_heads_kv * self%value_length, c_size_t) * &
            int(storage_size(self%x(1)) / 8, c_size_t)
        core_bytes = int(max(self%recurrent_inner_size, self%attention_heads * self%value_length), c_size_t) * &
            int(storage_size(self%x(1)) / 8, c_size_t)
        call self%cuda%allocate_buffer(qkv_bytes, self%cuda_qkv_device, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(ffn_bytes, self%cuda_gate_device, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(int(self%recurrent_value_heads, c_size_t) * 4_c_size_t, &
            self%cuda_alpha_device, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(int(self%recurrent_value_heads, c_size_t) * 4_c_size_t, &
            self%cuda_beta_device, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(ffn_bytes, self%cuda_ffn_gate_device, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(ffn_bytes, self%cuda_ffn_up_device, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(query_bytes, self%cuda_attention_q_device, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(key_bytes, self%cuda_attention_k_device, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(value_bytes, self%cuda_attention_v_device, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(core_bytes, self%cuda_attention_work_device, stat)
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
        ! GGML Q4 operations use their own CUDA scheduler stream.  Keep the
        ! resident mixed-quant pipeline deterministic; a graph capture of the
        ! Q8 stream cannot capture work submitted to that second scheduler.
        if (self%cuda_q4_resident) self%cuda_graph_enabled = .false.
        call release_host_attention_caches(self)
    end subroutine setup_cuda_device_pipeline

    subroutine release_host_attention_caches(self)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer :: i

        if (.not. allocated(self%layers)) return
        do i = 1, size(self%layers)
            if (self%layers(i)%recurrent) cycle
            if (allocated(self%layers(i)%key_cache)) deallocate(self%layers(i)%key_cache)
            if (allocated(self%layers(i)%value_cache)) deallocate(self%layers(i)%value_cache)
            if (allocated(self%layers(i)%key_cache_q8)) deallocate(self%layers(i)%key_cache_q8)
            if (allocated(self%layers(i)%value_cache_q8)) deallocate(self%layers(i)%value_cache_q8)
            if (allocated(self%layers(i)%key_cache_q8_scales)) &
                deallocate(self%layers(i)%key_cache_q8_scales)
            if (allocated(self%layers(i)%value_cache_q8_scales)) &
                deallocate(self%layers(i)%value_cache_q8_scales)
        end do
    end subroutine release_host_attention_caches

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
        if (c_associated(self%cuda_qkv_device)) call self%cuda%free_buffer(self%cuda_qkv_device, stat)
        if (c_associated(self%cuda_gate_device)) call self%cuda%free_buffer(self%cuda_gate_device, stat)
        if (c_associated(self%cuda_alpha_device)) call self%cuda%free_buffer(self%cuda_alpha_device, stat)
        if (c_associated(self%cuda_beta_device)) call self%cuda%free_buffer(self%cuda_beta_device, stat)
        if (c_associated(self%cuda_ffn_gate_device)) call self%cuda%free_buffer(self%cuda_ffn_gate_device, stat)
        if (c_associated(self%cuda_ffn_up_device)) call self%cuda%free_buffer(self%cuda_ffn_up_device, stat)
        if (c_associated(self%cuda_attention_q_device)) call self%cuda%free_buffer(self%cuda_attention_q_device, stat)
        if (c_associated(self%cuda_attention_k_device)) call self%cuda%free_buffer(self%cuda_attention_k_device, stat)
        if (c_associated(self%cuda_attention_v_device)) call self%cuda%free_buffer(self%cuda_attention_v_device, stat)
        if (c_associated(self%cuda_attention_work_device)) call self%cuda%free_buffer(self%cuda_attention_work_device, stat)
        if (c_associated(self%cuda_output_norm)) call self%cuda%free_buffer(self%cuda_output_norm, stat)
        if (c_associated(self%cuda_logits)) call self%cuda%free_buffer(self%cuda_logits, stat)
    end subroutine cuda_device_pipeline_cleanup

    subroutine qwen35_cpu_close(self)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer :: i
        type(status_t) :: cuda_stat

        call fast_context_destroy(self)
        self%fast_gpu = .false.
        call cuda_device_pipeline_cleanup(self, cuda_stat)
        if (allocated(self%cuda_weights)) then
            if (allocated(self%layers)) then
                do i = 1, size(self%layers)
                    call self%layers(i)%cuda_recurrent%destroy(cuda_stat)
                    call self%layers(i)%cuda_attention%destroy(cuda_stat)
                end do
            end if
            call self%mtp_layer%cuda_recurrent%destroy(cuda_stat)
            call self%mtp_layer%cuda_attention%destroy(cuda_stat)
            do i = 1, size(self%cuda_weights)
                call self%cuda_weights(i)%destroy(cuda_stat)
            end do
            deallocate(self%cuda_weights)
        end if
        if (allocated(self%cuda_weight_device)) deallocate(self%cuda_weight_device)
        if (allocated(self%cuda_q4_weights)) then
            do i = 1, size(self%cuda_q4_weights)
                call self%cuda_q4_weights(i)%destroy(cuda_stat)
            end do
            deallocate(self%cuda_q4_weights)
        end if
        call self%cuda_q4%destroy(cuda_stat)
        call self%cuda_second%destroy(cuda_stat)
        call self%cuda%destroy(cuda_stat)
        self%cuda_enabled = .false.
        self%cuda_q4_resident = .false.
        self%cuda_q4_split = .false.
        self%cuda_q8_split = .false.
        self%cuda_q8_cpu_override = .false.
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
                if (allocated(self%layers(i)%key_cache_q8)) deallocate(self%layers(i)%key_cache_q8)
                if (allocated(self%layers(i)%value_cache_q8)) deallocate(self%layers(i)%value_cache_q8)
                if (allocated(self%layers(i)%key_cache_q8_scales)) &
                    deallocate(self%layers(i)%key_cache_q8_scales)
                if (allocated(self%layers(i)%value_cache_q8_scales)) &
                    deallocate(self%layers(i)%value_cache_q8_scales)
            end do
            deallocate (self % layers)
        end if
        if (allocated(self%mtp_layer%conv_state)) deallocate(self%mtp_layer%conv_state)
        if (allocated(self%mtp_layer%gdn_state)) deallocate(self%mtp_layer%gdn_state)
        if (allocated(self%mtp_layer%key_cache)) deallocate(self%mtp_layer%key_cache)
        if (allocated(self%mtp_layer%value_cache)) deallocate(self%mtp_layer%value_cache)
        if (allocated(self%mtp_layer%key_cache_q8)) deallocate(self%mtp_layer%key_cache_q8)
        if (allocated(self%mtp_layer%value_cache_q8)) deallocate(self%mtp_layer%value_cache_q8)
        if (allocated(self%mtp_layer%key_cache_q8_scales)) deallocate(self%mtp_layer%key_cache_q8_scales)
        if (allocated(self%mtp_layer%value_cache_q8_scales)) deallocate(self%mtp_layer%value_cache_q8_scales)
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
        if (allocated(self%attention_key_work)) deallocate(self%attention_key_work)
        if (allocated(self%attention_value_work)) deallocate(self%attention_value_work)
        if (allocated(self % ffn_gate_work)) deallocate (self % ffn_gate_work)
        if (allocated(self % ffn_up_work)) deallocate (self % ffn_up_work)
        if (allocated(self % conv_work)) deallocate (self % conv_work)
        if (allocated(self % beta_work)) deallocate (self % beta_work)
        if (allocated(self % alpha_work)) deallocate (self % alpha_work)
        if (allocated(self % logits)) deallocate (self % logits)
        if (allocated(self%mtp_target_hidden)) deallocate(self%mtp_target_hidden)
        if (allocated(self%mtp_pending_hidden)) deallocate(self%mtp_pending_hidden)
        if (allocated(self%mtp_embedding)) deallocate(self%mtp_embedding)
        if (allocated(self%mtp_concat)) deallocate(self%mtp_concat)
        if (allocated(self%mtp_logits)) deallocate(self%mtp_logits)
        if (allocated(self % quantized_input)) deallocate (self % quantized_input)
        if (allocated(self % quantized_scales)) deallocate (self % quantized_scales)
        call self % file % close()
        if (allocated(self%model_path)) deallocate(self%model_path)
        self % hidden_size = 0
        self % vocabulary_size = 0
        self % layer_count = 0
        self%mtp_available = .false.
        self%mtp_active = .false.
        self%mtp_last_pair_position = -1_int64
        self%mtp_last_target_position = -1_int64
        self%mtp_last_pair_token = -1_int64
        self%mtp_last_draft_token = -1_int64
        self%mtp_last_draft_match = .false.
        self%mtp_eh_proj = 0
        self%mtp_enorm = 0
        self%mtp_hnorm = 0
        self%mtp_embed_tokens = 0
        self%mtp_shared_head_norm = 0
        self%mtp_shared_head_head = 0
        self%mtp_output = 0
        self%persistent_openmp = .false.
        self%persistent_openmp_active = .false.
        self%cache_key_q8 = .false.
        self%cache_value_q8 = .false.
        self%flash_attention_enabled = .true.
        self%cache_type_k = 1_int32
        self%cache_type_v = 1_int32
    end subroutine qwen35_cpu_close

    subroutine qwen35_cpu_reset(self)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer :: i
        type(status_t) :: cuda_stat

        if (self%fast_enabled) call fast_context_reset(self)
        if (allocated(self % x)) self % x = 0.0_real32
        if (allocated(self % layers)) then
            do i = 1, size(self % layers)
                if (allocated(self % layers(i) % conv_state)) self % layers(i) % conv_state = 0.0_real32
                if (allocated(self % layers(i) % gdn_state)) self % layers(i) % gdn_state = 0.0_real32
                if (allocated(self % layers(i) % key_cache)) self % layers(i) % key_cache = 0.0_real32
                if (allocated(self%layers(i)%value_cache)) self%layers(i)%value_cache = 0.0_real32
                if (allocated(self%layers(i)%key_cache_q8)) self%layers(i)%key_cache_q8 = 0_int8
                if (allocated(self%layers(i)%value_cache_q8)) self%layers(i)%value_cache_q8 = 0_int8
                if (allocated(self%layers(i)%key_cache_q8_scales)) self%layers(i)%key_cache_q8_scales = 0.0_real32
                if (allocated(self%layers(i)%value_cache_q8_scales)) &
                    self%layers(i)%value_cache_q8_scales = 0.0_real32
                if (self%cuda_enabled) call self%layers(i)%cuda_recurrent%reset(cuda_stat)
                if (self%cuda_enabled) call self%layers(i)%cuda_attention%reset(cuda_stat)
            end do
        end if
        if (allocated(self%mtp_layer%key_cache)) self%mtp_layer%key_cache = 0.0_real32
        if (allocated(self%mtp_layer%value_cache)) self%mtp_layer%value_cache = 0.0_real32
        if (allocated(self%mtp_layer%key_cache_q8)) self%mtp_layer%key_cache_q8 = 0_int8
        if (allocated(self%mtp_layer%value_cache_q8)) self%mtp_layer%value_cache_q8 = 0_int8
        if (allocated(self%mtp_layer%key_cache_q8_scales)) self%mtp_layer%key_cache_q8_scales = 0.0_real32
        if (allocated(self%mtp_layer%value_cache_q8_scales)) self%mtp_layer%value_cache_q8_scales = 0.0_real32
        if (allocated(self%mtp_pending_hidden)) self%mtp_pending_hidden = 0.0_real32
        self%mtp_last_pair_position = -1_int64
        self%mtp_last_target_position = -1_int64
        self%mtp_last_pair_token = -1_int64
        self%mtp_last_draft_token = -1_int64
        self%mtp_last_draft_match = .false.
    end subroutine qwen35_cpu_reset

    subroutine qwen35_cpu_forward(self, token_id, position, logits, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id, position
        real(real32), contiguous, intent(out) :: logits(:)
        type(status_t), intent(out) :: stat

        if (self%fast_enabled) then
            call qwen35_cpu_forward_body(self, token_id, position, logits, stat)
        else if (self%persistent_openmp .and. .not. self%cuda_device_pipeline) then
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

    logical function native_mtp_requested()
        character(len=64) :: value
        integer :: length

        native_mtp_requested = .false.
        value = ''
        call get_environment_variable('FORTAI_NATIVE_MTP', value, length=length)
        if (length > 0) then
            select case (trim(value(1:length)))
            case ('1', 'true', 'on', 'yes')
                native_mtp_requested = .true.
                return
            case ('0', 'false', 'off', 'no')
                return
            end select
        end if
        value = ''
        call get_environment_variable('FORTAI_SPEC_TYPE', value, length=length)
        if (length > 0) then
            if (trim(value(1:length)) == 'draft-mtp') then
                native_mtp_requested = .true.
                return
            end if
        end if
        value = ''
        call get_environment_variable('LLAMA_ARG_SPEC_TYPE', value, length=length)
        if (length > 0) then
            if (trim(value(1:length)) == 'draft-mtp') then
                native_mtp_requested = .true.
                return
            end if
        end if
        value = ''
        call get_environment_variable('LLAMACPP_SPEC_TYPE', value, length=length)
        if (length > 0) then
            native_mtp_requested = trim(value(1:length)) == 'draft-mtp'
            if (native_mtp_requested) return
        end if
        value = ''
        call get_environment_variable('FORTAI_DRAFT_MODEL', value, length=length)
        if (length > 0) then
            ! The published Qwen3.8 sidecar is a tensor carrier, not a
            ! standalone transformer.  Its filename is the only portable
            ! signal available to the native CLI when --model-draft is used.
            native_mtp_requested = index(value(1:length), 'mtp') > 0
        end if
    end function native_mtp_requested

    subroutine validate_mtp_sidecar(stat)
        type(status_t), intent(out) :: stat
        character(len=4096) :: path
        integer :: length
        logical :: exists

        call stat%clear()
        path = ''
        call get_environment_variable('FORTAI_DRAFT_MODEL', path, length=length)
        if (length <= 0) call get_environment_variable('LLAMA_ARG_MODEL_DRAFT', path, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_DRAFT_MODEL', path, length=length)
        if (length <= 0) return
        if (length > len(path)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 MTP sidecar path is too long')
            return
        end if
        inquire(file=trim(path(:length)), exist=exists)
        if (.not. exists) call stat%set(FORTAI_INVALID, 'Qwen3.5 MTP sidecar does not exist: ' // trim(path(:length)))
    end subroutine validate_mtp_sidecar

    integer function native_mtp_limit()
        character(len=32) :: value
        integer :: length, ios, parsed

        native_mtp_limit = 2
        value = ''
        call get_environment_variable('FORTAI_SPEC_DRAFT_N_MAX', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMA_ARG_SPEC_DRAFT_N_MAX', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_SPEC_DRAFT_N_MAX', value, length=length)
        if (length <= 0) return
        if (length > len(value)) return
        read(value(:length), *, iostat=ios) parsed
        if (ios == 0 .and. parsed >= 1 .and. parsed <= 32) native_mtp_limit = parsed
    end function native_mtp_limit

    subroutine qwen35_cpu_forward_greedy(self, token_id, position, next_token, logit_sum, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id, position
        integer(int64), intent(out) :: next_token
        real(real32), intent(out) :: logit_sum
        type(status_t), intent(out) :: stat
        integer(c_int) :: code, next_token_c

        next_token = 0_int64
        logit_sum = 0.0_real32
        if (self%fast_enabled) then
            code = fortai_llama_fast_context_decode_greedy(self%fast_handle, &
                int(token_id, c_int), int(position, c_int), next_token_c, logit_sum)
            if (code /= 0_c_int) then
                call stat%set(FORTAI_UNSUPPORTED, 'llama.cpp fast path greedy decode failed')
                return
            end if
            next_token = int(next_token_c, int64)
            call stat%clear()
            return
        end if

        if (.not. allocated(self%logits)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 logits workspace is not allocated')
            return
        end if
        if (self%cuda_device_pipeline) then
            call qwen35_cpu_forward_body(self, token_id, position, self%logits, stat, .false.)
            if (.not. stat%is_ok()) return
            block
                integer :: index
                call self%cuda%argmax_device(self%cuda_logits, int(self%vocabulary_size, c_size_t), index, stat)
                if (.not. stat%is_ok()) return
                next_token = int(index, int64)
            end block
            return
        end if
        call qwen35_cpu_forward(self, token_id, position, self%logits, stat)
        if (.not. stat%is_ok()) return
        logit_sum = sum(self%logits)
        next_token = int(maxloc(self%logits, dim=1) - 1, int64)
    end subroutine qwen35_cpu_forward_greedy

    subroutine qwen35_cpu_forward_greedy_speculative(self, token_id, position, tokens, count, logit_sum, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id, position
        integer(int64), intent(out) :: tokens(:)
        integer, intent(out) :: count
        real(real32), intent(out) :: logit_sum
        type(status_t), intent(out) :: stat
        integer(c_int) :: code, count_c
        integer(c_int) :: c_tokens(32)
        integer :: i, capacity
        integer(int64) :: draft_token, verified_token
        real(real32) :: draft_sum, verified_sum

        call stat%clear()
        count = 0
        logit_sum = 0.0_real32
        if (size(tokens) == 0) then
            call stat%set(FORTAI_INVALID, 'speculative output workspace is empty')
            return
        end if
        capacity = min(size(tokens), size(c_tokens))
        if (.not. self%fast_enabled .and. self%mtp_active .and. native_mtp_limit() >= 2 .and. &
            capacity >= 2 .and. position + 1_int64 < self%max_context) then
            call self%forward_greedy(token_id, position, tokens(1), logit_sum, stat)
            if (.not. stat%is_ok()) return
            call self%mtp_draft_greedy(tokens(1), position + 1_int64, draft_token, draft_sum, stat)
            if (.not. stat%is_ok()) return
            call self%forward_greedy(tokens(1), position + 1_int64, verified_token, verified_sum, stat)
            if (.not. stat%is_ok()) return
            tokens(2) = verified_token
            self%mtp_last_draft_token = draft_token
            self%mtp_last_draft_match = draft_token == verified_token
            count = 2
            return
        end if
        if (.not. self%fast_enabled) then
            call self%forward_greedy(token_id, position, tokens(1), logit_sum, stat)
            if (stat%is_ok()) count = 1
            return
        end if
        code = fortai_llama_fast_context_decode_speculative(self%fast_handle, &
            int(token_id, c_int), int(position, c_int), c_tokens, int(capacity, c_int), &
            count_c, logit_sum)
        if (code /= 0_c_int) then
            call stat%set(FORTAI_UNSUPPORTED, 'llama.cpp speculative decode failed')
            return
        end if
        count = int(count_c)
        if (count <= 0 .or. count > capacity) then
            call stat%set(FORTAI_UNSUPPORTED, 'llama.cpp speculative decode returned invalid count')
            count = 0
            return
        end if
        do i = 1, count
            tokens(i) = int(c_tokens(i), int64)
        end do
    end subroutine qwen35_cpu_forward_greedy_speculative

    subroutine qwen35_cpu_forward_body(self, token_id, position, logits, stat, download_logits)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id, position
        real(real32), contiguous, intent(out) :: logits(:)
        type(status_t), intent(out) :: stat
        logical, intent(in), optional :: download_logits
        integer :: i
        logical :: capture_graph, should_download

        call stat % clear()
        should_download = .true.
        if (present(download_logits)) should_download = download_logits
        if ((.not. self%fast_enabled .and. .not. allocated(self % x)) .or. &
            size(logits) /= self % vocabulary_size) then
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
        if (self%mtp_active) then
            if (.not. allocated(self%mtp_pending_hidden)) then
                call stat%set(FORTAI_INVALID, 'Qwen3.5 native MTP workspace is not allocated')
                return
            end if
            if (position == 0_int64) then
                if (self%mtp_last_target_position /= -1_int64) then
                    call stat%set(FORTAI_INVALID, 'Qwen3.5 native MTP position restarted without reset')
                    return
                end if
            else
                if (position /= self%mtp_last_target_position + 1_int64) then
                    call stat%set(FORTAI_INVALID, 'Qwen3.5 native MTP requires sequential positions')
                    return
                end if
                call qwen35_cpu_mtp_pair(self, token_id, position, self%mtp_pending_hidden, stat)
                if (.not. stat%is_ok()) return
            end if
        end if
        if (self%fast_enabled) then
            if (fortai_llama_fast_context_decode(self%fast_handle, int(token_id, c_int), &
                    int(position, c_int), logits, int(size(logits), c_size_t)) /= 0_c_int) then
                call stat%set(FORTAI_UNSUPPORTED, 'llama.cpp fast path decode failed')
            end if
            return
        end if
        if (self%cuda_device_pipeline) then
            call cuda_device_embedding(self, token_id, self%cuda_x, int(size(self%x), c_size_t), stat)
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
                    if (.not. stat%is_ok()) then
                        ! Preserve service availability if a CUDA scratch
                        ! allocation or launch is transiently unavailable.
                        ! The independent Q8 CPU path remains the oracle.
                        call model_matvec_pair(self, self%layers(i)%ffn_gate, self%layers(i)%ffn_up, &
                            self%normalized, self%ffn_gate_work, self%ffn_up_work, stat)
                        if (stat%is_ok()) then
                            call silu_product(self%ffn_gate_work, self%ffn_up_work)
                            call model_matvec(self, self%layers(i)%ffn_down, self%ffn_gate_work, &
                                self%hidden_work, stat)
                        end if
                    end if
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
            if (should_download) then
                call self%cuda%download_real(self%cuda_logits, logits, stat)
                if (.not. stat%is_ok()) return
            end if
            return
        end if
        call rms_norm(self%x, self%file%tensors(self%output_norm), self%norm_epsilon, &
            self % normalized, stat)
        if (.not. stat % is_ok()) return
        call model_matvec(self, self % output, self % normalized, logits, stat)
        if (stat%is_ok() .and. self%mtp_active) then
            self%mtp_pending_hidden = self%x
            self%mtp_last_target_position = position
        end if
    end subroutine qwen35_cpu_forward_body

    subroutine qwen35_cpu_mtp_pair(self, token_id, position, hidden, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id, position
        real(real32), contiguous, intent(in) :: hidden(:)
        type(status_t), intent(out) :: stat

        call stat%clear()
        if (.not. self%mtp_available .or. .not. self%mtp_active) then
            call stat%set(FORTAI_UNSUPPORTED, 'Qwen3.5 native MTP is not active')
            return
        end if
        if (self%mtp_last_pair_position == position .and. self%mtp_last_pair_token == token_id) return
        if (size(hidden) /= self%hidden_size .or. token_id < 0_int64 .or. &
            token_id >= self%vocabulary_size .or. position < 0_int64 .or. &
            position >= self%max_context) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 native MTP pair dimensions are invalid')
            return
        end if
        if (self%cache_key_q8) then
            if (.not. allocated(self%mtp_layer%key_cache_q8)) then
                call stat%set(FORTAI_INVALID, 'Qwen3.5 native MTP K q8 cache is not allocated')
                return
            end if
        else if (.not. allocated(self%mtp_layer%key_cache)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 native MTP K cache is not allocated')
            return
        end if
        if (self%cache_value_q8) then
            if (.not. allocated(self%mtp_layer%value_cache_q8)) then
                call stat%set(FORTAI_INVALID, 'Qwen3.5 native MTP V q8 cache is not allocated')
                return
            end if
        else if (.not. allocated(self%mtp_layer%value_cache)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 native MTP V cache is not allocated')
            return
        end if

        self%mtp_target_hidden = hidden
        if (self%mtp_embed_tokens > 0) then
            call self%file%tensors(self%mtp_embed_tokens)%get_row(token_id + 1_int64, &
                self%mtp_embedding, stat)
        else
            call self%file%tensors(self%token_embedding)%get_row(token_id + 1_int64, &
                self%mtp_embedding, stat)
        end if
        if (.not. stat%is_ok()) return
        call rms_norm(self%mtp_embedding, self%file%tensors(self%mtp_enorm), self%norm_epsilon, &
            self%normalized, stat)
        if (.not. stat%is_ok()) return
        call rms_norm(self%mtp_target_hidden, self%file%tensors(self%mtp_hnorm), self%norm_epsilon, &
            self%hidden_work, stat)
        if (.not. stat%is_ok()) return
        self%mtp_concat(1:self%hidden_size) = self%normalized
        self%mtp_concat(self%hidden_size + 1:2 * self%hidden_size) = self%hidden_work
        call model_matvec(self, self%mtp_eh_proj, self%mtp_concat, self%x, stat)
        if (.not. stat%is_ok()) return

        self%residual = self%x
        call rms_norm(self%x, self%file%tensors(self%mtp_layer%attn_norm), self%norm_epsilon, &
            self%normalized, stat)
        if (.not. stat%is_ok()) return
        call forward_attention(self, self%mtp_layer, position, stat)
        if (.not. stat%is_ok()) return
        self%x = self%hidden_work + self%residual
        self%residual = self%x
        call rms_norm(self%x, self%file%tensors(self%mtp_layer%post_norm), self%norm_epsilon, &
            self%normalized, stat)
        if (.not. stat%is_ok()) return
        call model_matvec_pair(self, self%mtp_layer%ffn_gate, self%mtp_layer%ffn_up, &
            self%normalized, self%ffn_gate_work, self%ffn_up_work, stat)
        if (.not. stat%is_ok()) return
        call silu_product(self%ffn_gate_work, self%ffn_up_work)
        call model_matvec(self, self%mtp_layer%ffn_down, self%ffn_gate_work, self%hidden_work, stat)
        if (.not. stat%is_ok()) return
        self%x = self%hidden_work + self%residual
        call rms_norm(self%x, self%file%tensors(self%mtp_shared_head_norm), self%norm_epsilon, &
            self%normalized, stat)
        if (.not. stat%is_ok()) return
        call model_matvec(self, self%mtp_output, self%normalized, self%mtp_logits, stat)
        if (.not. stat%is_ok()) return

        self%x = self%mtp_target_hidden
        self%mtp_last_pair_position = position
        self%mtp_last_pair_token = token_id
    end subroutine qwen35_cpu_mtp_pair

    subroutine qwen35_cpu_mtp_draft_greedy(self, token_id, position, next_token, logit_sum, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id, position
        integer(int64), intent(out) :: next_token
        real(real32), intent(out) :: logit_sum
        type(status_t), intent(out) :: stat

        next_token = 0_int64
        logit_sum = 0.0_real32
        call stat%clear()
        if (.not. self%mtp_active) then
            call stat%set(FORTAI_UNSUPPORTED, 'Qwen3.5 native MTP is not active')
            return
        end if
        if (.not. allocated(self%mtp_logits)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 native MTP logits workspace is not allocated')
            return
        end if
        if (.not. allocated(self%mtp_pending_hidden)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 native MTP hidden workspace is not allocated')
            return
        end if
        if (position <= 0_int64 .or. &
            position /= self%mtp_last_target_position + 1_int64) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 native MTP draft requires the next position')
            return
        end if
        call qwen35_cpu_mtp_pair(self, token_id, position, self%mtp_pending_hidden, stat)
        if (.not. stat%is_ok()) return
        next_token = int(maxloc(self%mtp_logits, dim=1) - 1, int64)
        logit_sum = sum(self%mtp_logits)
    end subroutine qwen35_cpu_mtp_draft_greedy

    integer function fast_path_mode()
        character(len=16) :: value
        integer :: length

        ! Prefer the resident llama.cpp graph when it is available.  The
        ! native Fortran/GGML implementation remains an automatic fallback;
        ! callers can force it with FORTAI_LLAMA_FASTPATH=native|0.
        fast_path_mode = 3
        value = ''
        call get_environment_variable('FORTAI_LLAMA_FASTPATH', value, length=length)
        if (length <= 0) return
        if (length > len(value)) length = len(value)
        select case (trim(value(1:length)))
        case ('0', 'native', 'off', 'none')
            fast_path_mode = 0
        case ('1', 'auto')
            fast_path_mode = 3
        case ('cpu')
            fast_path_mode = 1
        case ('cuda', 'gpu')
            fast_path_mode = 2
        end select
    end function fast_path_mode

    subroutine fast_context_create(self, gpu_layers, main_gpu, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: gpu_layers, main_gpu
        type(status_t), intent(out) :: stat
        character(len=32) :: thread_text
        character(kind=c_char), allocatable :: cpath(:)
        integer :: length, i, threads, ios
        integer(c_int) :: code, vocabulary, layers

        call stat%clear()
        call fast_context_destroy(self)
        if (.not. allocated(self%model_path) .or. len_trim(self%model_path) == 0) then
            call stat%set(FORTAI_UNSUPPORTED, 'llama.cpp fast path has no model path')
            return
        end if
        if (fortai_llama_fast_available() == 0_c_int) then
            call stat%set(FORTAI_UNSUPPORTED, 'llama.cpp fast path library is unavailable')
            return
        end if
        threads = 1
        thread_text = ''
        call get_environment_variable('OMP_NUM_THREADS', thread_text, length=length)
        if (length > 0) then
            read (thread_text(1:min(length, len(thread_text))), *, iostat=ios) threads
            if (ios /= 0 .or. threads <= 0) threads = 1
        end if
        length = len_trim(self%model_path)
        allocate(cpath(length + 1))
        do i = 1, length
            cpath(i) = self%model_path(i:i)
        end do
        cpath(length + 1) = c_null_char
        code = fortai_llama_fast_context_create(cpath, int(self%max_context, c_int), int(threads, c_int), &
            int(gpu_layers, c_int), int(main_gpu, c_int), self%fast_handle, vocabulary, layers)
        deallocate(cpath)
        if (code /= 0_c_int) then
            self%fast_handle = c_null_ptr
            call stat%set(FORTAI_UNSUPPORTED, 'llama.cpp fast path context creation failed')
            return
        end if
        self%vocabulary_size = int(vocabulary, int32)
        self%layer_count = int(layers, int32)
        self%fast_enabled = .true.
    end subroutine fast_context_create

    subroutine fast_context_destroy(self)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(c_int) :: code

        if (c_associated(self%fast_handle)) then
            code = fortai_llama_fast_context_destroy(self%fast_handle)
            self%fast_handle = c_null_ptr
        end if
        self%fast_enabled = .false.
    end subroutine fast_context_destroy

    subroutine fast_context_reset(self)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(c_int) :: code

        if (c_associated(self%fast_handle)) code = fortai_llama_fast_context_reset(self%fast_handle)
    end subroutine fast_context_reset

    subroutine enable_persistent_openmp(self)
        class(qwen35_cpu_model_t), intent(inout) :: self
        character(len=8) :: enabled
        integer :: length

        call get_environment_variable('FORTAI_ENABLE_PERSISTENT_OPENMP', enabled, length=length)
        if (length > 0) then
            if (enabled(1:length) == '1') self%persistent_openmp = .true.
        end if
    end subroutine enable_persistent_openmp

    subroutine cuda_device_matvec(self, tensor_index, device_input, input_elements, device_output, &
            output_elements, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: tensor_index
        type(c_ptr), intent(in) :: device_input, device_output
        integer(c_size_t), intent(in) :: input_elements, output_elements
        type(status_t), intent(out) :: stat

        call stat%clear()
        if (self%file%tensors(tensor_index)%value_type == GGML_TYPE_Q8_0) then
            call cuda_q8_matvec_device_f32(self%cuda, self%cuda_weights(tensor_index), device_input, &
                input_elements, device_output, output_elements, stat)
        else if (self%cuda_q4_resident .and. allocated(self%cuda_q4_weights) .and. &
                c_associated(self%cuda_q4_weights(tensor_index)%handle)) then
            ! Q4's GGML scheduler has a separate stream.  Complete the Q8
            ! producer before handing its pointer to that backend; the Q4
            ! call synchronizes its own stream before returning.
            call self%cuda%synchronize(stat)
            if (.not. stat%is_ok()) return
            call cuda_q4_matvec_device(self%cuda_q4, self%cuda_q4_weights(tensor_index), device_input, &
                input_elements, device_output, output_elements, stat)
        else
            call stat%set(FORTAI_UNSUPPORTED, 'quantized tensor is not device resident')
        end if
    end subroutine cuda_device_matvec

    subroutine cuda_device_matvec_pair(self, first_index, second_index, device_input, input_elements, &
            first_output, first_elements, second_output, second_elements, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: first_index, second_index
        type(c_ptr), intent(in) :: device_input, first_output, second_output
        integer(c_size_t), intent(in) :: input_elements, first_elements, second_elements
        type(status_t), intent(out) :: stat

        call stat%clear()
        if (is_q4_xl_type(self%file%tensors(first_index)%value_type) .and. &
                is_q4_xl_type(self%file%tensors(second_index)%value_type) .and. &
                self%cuda_q4_resident .and. self%cuda_q4_group_enabled .and. allocated(self%cuda_q4_weights) .and. &
                c_associated(self%cuda_q4_weights(first_index)%handle) .and. &
                c_associated(self%cuda_q4_weights(second_index)%handle)) then
            call self%cuda%synchronize(stat)
            if (.not. stat%is_ok()) return
            call cuda_q4_matvec_device_pair(self%cuda_q4, self%cuda_q4_weights(first_index), &
                self%cuda_q4_weights(second_index), device_input, input_elements, first_output, first_elements, &
                second_output, second_elements, stat)
            return
        end if
        call cuda_device_matvec(self, first_index, device_input, input_elements, first_output, first_elements, stat)
        if (.not. stat%is_ok()) return
        call cuda_device_matvec(self, second_index, device_input, input_elements, second_output, second_elements, stat)
    end subroutine cuda_device_matvec_pair

    subroutine cuda_device_matvec_triplet(self, first_index, second_index, third_index, device_input, &
            input_elements, first_output, first_elements, second_output, second_elements, third_output, &
            third_elements, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: first_index, second_index, third_index
        type(c_ptr), intent(in) :: device_input, first_output, second_output, third_output
        integer(c_size_t), intent(in) :: input_elements, first_elements, second_elements, third_elements
        type(status_t), intent(out) :: stat

        call stat%clear()
        if (is_q4_xl_type(self%file%tensors(first_index)%value_type) .and. &
                is_q4_xl_type(self%file%tensors(second_index)%value_type) .and. &
                is_q4_xl_type(self%file%tensors(third_index)%value_type) .and. self%cuda_q4_resident .and. &
                self%cuda_q4_group_enabled .and. &
                allocated(self%cuda_q4_weights) .and. c_associated(self%cuda_q4_weights(first_index)%handle) .and. &
                c_associated(self%cuda_q4_weights(second_index)%handle) .and. &
                c_associated(self%cuda_q4_weights(third_index)%handle)) then
            call self%cuda%synchronize(stat)
            if (.not. stat%is_ok()) return
            call cuda_q4_matvec_device_triplet(self%cuda_q4, self%cuda_q4_weights(first_index), &
                self%cuda_q4_weights(second_index), self%cuda_q4_weights(third_index), device_input, &
                input_elements, first_output, first_elements, second_output, second_elements, third_output, &
                third_elements, stat)
            return
        end if
        call cuda_device_matvec(self, first_index, device_input, input_elements, first_output, first_elements, stat)
        if (.not. stat%is_ok()) return
        call cuda_device_matvec(self, second_index, device_input, input_elements, second_output, second_elements, stat)
        if (.not. stat%is_ok()) return
        call cuda_device_matvec(self, third_index, device_input, input_elements, third_output, third_elements, stat)
    end subroutine cuda_device_matvec_triplet

    subroutine cuda_device_embedding(self, token_id, device_output, output_elements, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id
        type(c_ptr), intent(in) :: device_output
        integer(c_size_t), intent(in) :: output_elements
        type(status_t), intent(out) :: stat

        call stat%clear()
        if (self%file%tensors(self%token_embedding)%value_type == GGML_TYPE_Q8_0) then
            call cuda_qwen35_embedding_device(self%cuda, self%cuda_weights(self%token_embedding), &
                int(token_id, c_int64_t), device_output, output_elements, stat)
        else if (self%cuda_q4_resident) then
            call cuda_q4_embedding_device(self%cuda_q4, self%cuda_q4_weights(self%token_embedding), &
                int(token_id, c_int64_t), device_output, output_elements, stat)
        else
            call stat%set(FORTAI_UNSUPPORTED, 'token embedding is not device resident')
        end if
    end subroutine cuda_device_embedding

    subroutine forward_ffn_device(self, layer_index, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: layer_index
        type(status_t), intent(out) :: stat
        integer(c_size_t) :: hidden_elements, ffn_elements

        call stat%clear()
        hidden_elements = int(size(self%x), c_size_t)
        ffn_elements = int(self%feed_forward_size, c_size_t)
        if (cuda_ffn_ready(self, self%layers(layer_index))) then
            call cuda_q8_ffn_device(self%cuda, self%cuda_weights(self%layers(layer_index)%ffn_gate), &
                self%cuda_weights(self%layers(layer_index)%ffn_up), &
                self%cuda_weights(self%layers(layer_index)%ffn_down), self%cuda_normalized, &
                hidden_elements, self%cuda_hidden, hidden_elements, stat)
            return
        end if
        call cuda_device_matvec_pair(self, self%layers(layer_index)%ffn_gate, self%layers(layer_index)%ffn_up, &
            self%cuda_normalized, hidden_elements, self%cuda_ffn_gate_device, ffn_elements, &
            self%cuda_ffn_up_device, ffn_elements, stat)
        if (.not. stat%is_ok()) return
        call cuda_qwen35_silu_product_device(self%cuda, self%cuda_ffn_gate_device, &
            self%cuda_ffn_up_device, ffn_elements, stat)
        if (.not. stat%is_ok()) return
        call cuda_device_matvec(self, self%layers(layer_index)%ffn_down, self%cuda_ffn_gate_device, &
            ffn_elements, self%cuda_hidden, hidden_elements, stat)
    end subroutine forward_ffn_device

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
        if (all_q8_recurrent_weights(self, layer_index)) then
            call self%layers(layer_index)%cuda_recurrent%run_device(self%cuda_normalized, hidden_elements, &
                self%cuda_hidden, hidden_elements, stat)
        else
            call cuda_device_matvec_pair(self, self%layers(layer_index)%attn_qkv, &
                self%layers(layer_index)%attn_gate, self%cuda_normalized, hidden_elements, self%cuda_qkv_device, &
                int(self%recurrent_conv_size, c_size_t), self%cuda_gate_device, &
                int(self%recurrent_inner_size, c_size_t), stat)
            if (.not. stat%is_ok()) return
            call cuda_device_matvec_pair(self, self%layers(layer_index)%ssm_alpha, &
                self%layers(layer_index)%ssm_beta, self%cuda_normalized, hidden_elements, self%cuda_alpha_device, &
                int(self%recurrent_value_heads, c_size_t), self%cuda_beta_device, &
                int(self%recurrent_value_heads, c_size_t), stat)
            if (.not. stat%is_ok()) return
            call self%layers(layer_index)%cuda_recurrent%run_core_device(self%cuda_qkv_device, &
                int(self%recurrent_conv_size, c_size_t), self%cuda_gate_device, &
                int(self%recurrent_inner_size, c_size_t), self%cuda_alpha_device, &
                int(self%recurrent_value_heads, c_size_t), self%cuda_beta_device, &
                int(self%recurrent_value_heads, c_size_t), self%cuda_attention_work_device, &
                int(self%recurrent_inner_size, c_size_t), stat)
            if (.not. stat%is_ok()) return
            call cuda_device_matvec(self, self%layers(layer_index)%ssm_out, self%cuda_attention_work_device, &
                int(self%recurrent_inner_size, c_size_t), self%cuda_hidden, hidden_elements, stat)
        end if
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
        call forward_ffn_device(self, layer_index, stat)
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
        call cuda_device_matvec(self, self%output, self%cuda_normalized, hidden_elements, &
            self%cuda_logits, logits_elements, stat)
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
        if (all_q8_attention_weights(self, layer_index)) then
            call self%layers(layer_index)%cuda_attention%run_device(self%cuda_normalized, hidden_elements, &
                int(position), self%cuda_hidden, hidden_elements, stat)
        else
            call cuda_device_matvec_triplet(self, self%layers(layer_index)%attn_q, &
                self%layers(layer_index)%attn_k, self%layers(layer_index)%attn_v, self%cuda_normalized, hidden_elements, &
                self%cuda_attention_q_device, int(2 * self%attention_heads * self%attention_head_size, c_size_t), &
                self%cuda_attention_k_device, int(self%attention_heads_kv * self%attention_head_size, c_size_t), &
                self%cuda_attention_v_device, int(self%attention_heads_kv * self%value_length, c_size_t), stat)
            if (.not. stat%is_ok()) return
            call self%layers(layer_index)%cuda_attention%run_core_device(self%cuda_attention_q_device, &
                int(2 * self%attention_heads * self%attention_head_size, c_size_t), self%cuda_attention_k_device, &
                int(self%attention_heads_kv * self%attention_head_size, c_size_t), self%cuda_attention_v_device, &
                int(self%attention_heads_kv * self%value_length, c_size_t), int(position), &
                self%cuda_attention_work_device, int(self%attention_heads * self%value_length, c_size_t), stat)
            if (.not. stat%is_ok()) return
            call cuda_device_matvec(self, self%layers(layer_index)%attn_out, self%cuda_attention_work_device, &
                int(self%attention_heads * self%value_length, c_size_t), self%cuda_hidden, hidden_elements, stat)
        end if
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
        call forward_ffn_device(self, layer_index, stat)
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
        ! A state-only recurrent handle is used when one or more projection
        ! matrices are Q4.  It owns the GDN/conv state but has no Q8 weight
        ! bundle, so only the full Q8 handle may enter the host `run` path.
        if (self%cuda_enabled .and. self%cuda_device_pipeline .and. &
                all_q8_recurrent_weights(self, layer_index) .and. &
                c_associated(self%layers(layer_index)%cuda_recurrent%handle)) then
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
        self%cuda_q8_cpu_override = .true.
        if (cuda_recurrent_projection_ready(self, layer)) then
            if (mod(size(input), 32) /= 0) then
                call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA Q8 input is not block aligned')
                self%cuda_q8_cpu_override = .false.
                return
            end if
            call fortai_q8_quantize(input, self%quantized_input, self%quantized_scales, &
                int(size(input), c_int64_t))
            call cuda_model_matvec_pair_quantized(self, layer%attn_qkv, layer%attn_gate, size(input), &
                self%qkv_work(1:self%recurrent_conv_size), self%gate_work(1:self%recurrent_inner_size), stat)
            if (.not. stat%is_ok()) then
                self%cuda_q8_cpu_override = .false.
                return
            end if
            call cuda_model_matvec_pair_quantized(self, layer%ssm_alpha, layer%ssm_beta, size(input), &
                self%alpha_work, self%beta_work, stat)
            if (.not. stat%is_ok()) then
                self%cuda_q8_cpu_override = .false.
                return
            end if
        else
            call model_matvec_pair(self, layer%attn_qkv, layer%attn_gate, input, &
                self%qkv_work(1:self%recurrent_conv_size), self%gate_work(1:self%recurrent_inner_size), stat)
            if (.not. stat % is_ok()) then
                self%cuda_q8_cpu_override = .false.
                return
            end if
            call model_matvec_pair(self, layer%ssm_alpha, layer%ssm_beta, input, &
                self%alpha_work, self%beta_work, stat)
            if (.not. stat % is_ok()) then
                self%cuda_q8_cpu_override = .false.
                return
            end if
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
        self%cuda_q8_cpu_override = .false.
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
        if (self%layer_count > 64_int32 .or. self%mtp_active) return
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

    logical function cuda_quantized_device_ready(self, tensor_index)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: tensor_index

        cuda_quantized_device_ready = .false.
        if (tensor_index <= 0 .or. tensor_index > size(self%file%tensors)) return
        if (self%file%tensors(tensor_index)%value_type == GGML_TYPE_Q8_0) then
            if (.not. allocated(self%cuda_weights)) return
            if (tensor_index > size(self%cuda_weights)) return
            cuda_quantized_device_ready = c_associated(self%cuda_weights(tensor_index)%handle)
        else if (is_q4_xl_type(self%file%tensors(tensor_index)%value_type)) then
            if (.not. self%cuda_q4_resident) return
            if (.not. allocated(self%cuda_q4_weights)) return
            if (tensor_index > size(self%cuda_q4_weights)) return
            cuda_quantized_device_ready = c_associated(self%cuda_q4_weights(tensor_index)%handle)
        end if
    end function cuda_quantized_device_ready

    logical function cuda_recurrent_device_ready(self, layer_index)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: layer_index
        integer :: indices(5), i

        cuda_recurrent_device_ready = .false.
        if (layer_index <= 0 .or. layer_index > size(self%layers)) return
        indices = [self%layers(layer_index)%attn_qkv, self%layers(layer_index)%attn_gate, &
            self%layers(layer_index)%ssm_alpha, self%layers(layer_index)%ssm_beta, &
            self%layers(layer_index)%ssm_out]
        do i = 1, size(indices)
            if (.not. cuda_quantized_device_ready(self, indices(i))) return
        end do
        cuda_recurrent_device_ready = .true.
    end function cuda_recurrent_device_ready

    logical function cuda_attention_device_ready(self, layer_index)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: layer_index
        integer :: indices(4), i

        cuda_attention_device_ready = .false.
        if (layer_index <= 0 .or. layer_index > size(self%layers)) return
        indices = [self%layers(layer_index)%attn_q, self%layers(layer_index)%attn_k, &
            self%layers(layer_index)%attn_v, self%layers(layer_index)%attn_out]
        do i = 1, size(indices)
            if (.not. cuda_quantized_device_ready(self, indices(i))) return
        end do
        cuda_attention_device_ready = .true.
    end function cuda_attention_device_ready

    logical function cuda_recurrent_projection_ready(self, layer)
        class(qwen35_cpu_model_t), intent(in) :: self
        type(qwen35_cpu_layer_t), intent(in) :: layer

        cuda_recurrent_projection_ready = .false.
        if (.not. self%cuda_enabled) return
        if (self%cuda_q8_cpu_override) return
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
        if (self%cuda_q8_cpu_override) return
        if (.not. cuda_host_q8_enabled()) return
        if (.not. cuda_fused_ffn_host_enabled()) return
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

    logical function cuda_host_q8_enabled()
        character(len=8) :: value
        integer :: length

        cuda_host_q8_enabled = .true.
        value = ''
        call get_environment_variable('FORTAI_CUDA_Q8_HOST', value, length=length)
        if (length > 0) cuda_host_q8_enabled = value(1:min(length, len(value))) /= '0'
    end function cuda_host_q8_enabled

    logical function cuda_fused_ffn_host_enabled()
        character(len=8) :: value
        integer :: length

        cuda_fused_ffn_host_enabled = .false.
        value = ''
        call get_environment_variable('FORTAI_CUDA_Q8_FFN_FUSED', value, length=length)
        if (length > 0) cuda_fused_ffn_host_enabled = value(1:min(length, len(value))) == '1'
    end function cuda_fused_ffn_host_enabled

    logical function cuda_ffn_device_ready(self, layer)
        class(qwen35_cpu_model_t), intent(in) :: self
        type(qwen35_cpu_layer_t), intent(in) :: layer

        cuda_ffn_device_ready = .false.
        if (.not. cuda_quantized_device_ready(self, layer%ffn_gate)) return
        if (.not. cuda_quantized_device_ready(self, layer%ffn_up)) return
        if (.not. cuda_quantized_device_ready(self, layer%ffn_down)) return
        cuda_ffn_device_ready = .true.
    end function cuda_ffn_device_ready

    subroutine forward_attention(self, layer, position, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(qwen35_cpu_layer_t), intent(inout) :: layer
        integer(int64), intent(in) :: position
        type(status_t), intent(out) :: stat
        integer :: head, i, kv_head, q_offset, k_offset, v_offset

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
            q_offset = (head - 1) * 2 * self % attention_head_size
            v_offset = (head - 1) * self % value_length
            call run_attention(self, layer, head, kv_head, position, q_offset, v_offset)
            do i = 1, self % value_length
                self % attention_work(v_offset + i) = self % attention_work(v_offset + i) * &
                    sigmoid(self % q_work(q_offset_for_gate(head, self % attention_head_size) + i))
            end do
        end do
        call model_matvec(self, layer%attn_out, self%attention_work, self%hidden_work, stat)
    end subroutine forward_attention

    subroutine run_attention(self, layer, head, kv_head, position, q_offset, v_offset)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(qwen35_cpu_layer_t), intent(inout) :: layer
        integer, intent(in) :: head, kv_head, q_offset, v_offset
        integer(int64), intent(in) :: position
        integer(c_int64_t) :: key_stride, value_stride
        real(real32) :: attention_scale

        attention_scale = 1.0_real32 / sqrt(real(self%attention_head_size, real32))
        if (.not. self%flash_attention_enabled) then
            call scalar_attention(self, layer, head, kv_head, position, q_offset, v_offset, attention_scale)
            return
        end if
        key_stride = int(self%attention_head_size * self%attention_heads_kv, c_int64_t)
        value_stride = int(self%value_length * self%attention_heads_kv, c_int64_t)
        if (self%cache_key_q8) then
            call dequantize_attention_cache(self, layer, kv_head, position, .true., self%attention_key_work)
            key_stride = int(self%attention_head_size, c_int64_t)
        end if
        if (self%cache_value_q8) then
            call dequantize_attention_cache(self, layer, kv_head, position, .false., self%attention_value_work)
            value_stride = int(self%value_length, c_int64_t)
        end if
        if (self%cache_key_q8) then
            if (self%cache_value_q8) then
                call fortai_flash_attention_f16(self%q_work(q_offset + 1:), self%attention_key_work, &
                    self%attention_value_work, position + 1_int64, key_stride, value_stride, &
                    int(self%attention_head_size, c_int64_t), int(self%value_length, c_int64_t), &
                    attention_scale, self%attention_work(v_offset + 1:))
            else
                call fortai_flash_attention_f16(self%q_work(q_offset + 1:), self%attention_key_work, &
                    layer%value_cache(kv_head * self%value_length + 1:), position + 1_int64, key_stride, &
                    value_stride, int(self%attention_head_size, c_int64_t), int(self%value_length, c_int64_t), &
                    attention_scale, self%attention_work(v_offset + 1:))
            end if
        else if (self%cache_value_q8) then
            call fortai_flash_attention_f16(self%q_work(q_offset + 1:), &
                layer%key_cache(kv_head * self%attention_head_size + 1:), self%attention_value_work, &
                position + 1_int64, key_stride, value_stride, int(self%attention_head_size, c_int64_t), &
                int(self%value_length, c_int64_t), attention_scale, self%attention_work(v_offset + 1:))
        else
            call fortai_flash_attention_f16(self%q_work(q_offset + 1:), &
                layer%key_cache(kv_head * self%attention_head_size + 1:), &
                layer%value_cache(kv_head * self%value_length + 1:), position + 1_int64, key_stride, &
                value_stride, int(self%attention_head_size, c_int64_t), int(self%value_length, c_int64_t), &
                attention_scale, self%attention_work(v_offset + 1:))
        end if
    end subroutine run_attention

    subroutine scalar_attention(self, layer, head, kv_head, position, q_offset, v_offset, scale)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(qwen35_cpu_layer_t), intent(in) :: layer
        integer, intent(in) :: head, kv_head, q_offset, v_offset
        integer(int64), intent(in) :: position
        real(real32), intent(in) :: scale
        integer :: current_position, i, j, key_index, value_index
        real(real32) :: dot, maximum, normalizer, weight

        if (self%cache_key_q8) call dequantize_attention_cache(self, layer, kv_head, position, .true., &
            self%attention_key_work)
        if (self%cache_value_q8) call dequantize_attention_cache(self, layer, kv_head, position, .false., &
            self%attention_value_work)
        maximum = -huge(0.0_real32)
        do current_position = 0, int(position)
            dot = 0.0_real32
            do j = 1, self%attention_head_size
                if (self%cache_key_q8) then
                    key_index = current_position * self%attention_head_size + j
                    dot = dot + self%q_work(q_offset + j) * self%attention_key_work(key_index)
                else
                    key_index = current_position * self%attention_head_size * self%attention_heads_kv + &
                        kv_head * self%attention_head_size + j
                    dot = dot + self%q_work(q_offset + j) * layer%key_cache(key_index)
                end if
            end do
            self%scores(current_position + 1) = dot * scale
            maximum = max(maximum, self%scores(current_position + 1))
        end do
        normalizer = 0.0_real32
        do current_position = 0, int(position)
            self%scores(current_position + 1) = exp(self%scores(current_position + 1) - maximum)
            normalizer = normalizer + self%scores(current_position + 1)
        end do
        self%attention_work(v_offset + 1:v_offset + self%value_length) = 0.0_real32
        if (normalizer <= 0.0_real32) return
        do current_position = 0, int(position)
            weight = self%scores(current_position + 1) / normalizer
            do i = 1, self%value_length
                if (self%cache_value_q8) then
                    value_index = current_position * self%value_length + i
                    self%attention_work(v_offset + i) = self%attention_work(v_offset + i) + &
                        weight * self%attention_value_work(value_index)
                else
                    value_index = current_position * self%value_length * self%attention_heads_kv + &
                        kv_head * self%value_length + i
                    self%attention_work(v_offset + i) = self%attention_work(v_offset + i) + &
                        weight * layer%value_cache(value_index)
                end if
            end do
        end do
    end subroutine scalar_attention

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

    subroutine bind_mtp_layer(self, block_number, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: block_number
        type(status_t), intent(out) :: stat
        type(qwen35_cpu_layer_t) :: layer

        call stat%clear()
        layer%recurrent = .false.
        layer%attn_norm = find_layer_tensor(self%file, block_number, 'attn_norm.weight')
        layer%post_norm = find_layer_tensor(self%file, block_number, 'post_attention_norm.weight')
        layer%ffn_gate = find_layer_tensor(self%file, block_number, 'ffn_gate.weight')
        layer%ffn_up = find_layer_tensor(self%file, block_number, 'ffn_up.weight')
        layer%ffn_down = find_layer_tensor(self%file, block_number, 'ffn_down.weight')
        layer%attn_q = find_layer_tensor(self%file, block_number, 'attn_q.weight')
        layer%attn_k = find_layer_tensor(self%file, block_number, 'attn_k.weight')
        layer%attn_v = find_layer_tensor(self%file, block_number, 'attn_v.weight')
        layer%attn_out = find_layer_tensor(self%file, block_number, 'attn_output.weight')
        layer%q_norm = find_layer_tensor(self%file, block_number, 'attn_q_norm.weight')
        layer%k_norm = find_layer_tensor(self%file, block_number, 'attn_k_norm.weight')
        self%mtp_eh_proj = find_layer_tensor(self%file, block_number, 'nextn.eh_proj.weight')
        self%mtp_enorm = find_layer_tensor(self%file, block_number, 'nextn.enorm.weight')
        self%mtp_hnorm = find_layer_tensor(self%file, block_number, 'nextn.hnorm.weight')
        self%mtp_embed_tokens = find_layer_tensor(self%file, block_number, 'nextn.embed_tokens.weight')
        self%mtp_shared_head_norm = find_layer_tensor(self%file, block_number, &
            'nextn.shared_head_norm.weight')
        self%mtp_shared_head_head = find_layer_tensor(self%file, block_number, &
            'nextn.shared_head_head.weight')
        self%mtp_output = self%output
        if (self%mtp_shared_head_head > 0) self%mtp_output = self%mtp_shared_head_head

        if (layer%attn_norm == 0 .or. layer%post_norm == 0 .or. layer%ffn_gate == 0 .or. &
            layer%ffn_up == 0 .or. layer%ffn_down == 0 .or. layer%attn_q == 0 .or. &
            layer%attn_k == 0 .or. layer%attn_v == 0 .or. layer%attn_out == 0 .or. &
            layer%q_norm == 0 .or. layer%k_norm == 0 .or. self%mtp_eh_proj == 0 .or. &
            self%mtp_enorm == 0 .or. self%mtp_hnorm == 0) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 MTP block is incomplete')
            return
        end if
        if (self%mtp_shared_head_norm == 0) self%mtp_shared_head_norm = self%output_norm
        call check_tensor_shape(self, layer%attn_norm, 1, self%hidden_size, 0, stat)
        if (.not. stat%is_ok()) return
        call check_tensor_shape(self, layer%post_norm, 1, self%hidden_size, 0, stat)
        if (.not. stat%is_ok()) return
        call check_tensor_shape(self, layer%ffn_gate, 2, self%hidden_size, self%feed_forward_size, stat)
        if (.not. stat%is_ok()) return
        call check_tensor_shape(self, layer%ffn_up, 2, self%hidden_size, self%feed_forward_size, stat)
        if (.not. stat%is_ok()) return
        call check_tensor_shape(self, layer%ffn_down, 2, self%feed_forward_size, self%hidden_size, stat)
        if (.not. stat%is_ok()) return
        call check_tensor_shape(self, layer%attn_q, 2, self%hidden_size, &
            2 * self%attention_heads * self%attention_head_size, stat)
        if (.not. stat%is_ok()) return
        call check_tensor_shape(self, layer%attn_k, 2, self%hidden_size, &
            self%attention_heads_kv * self%attention_head_size, stat)
        if (.not. stat%is_ok()) return
        call check_tensor_shape(self, layer%attn_v, 2, self%hidden_size, &
            self%attention_heads_kv * self%value_length, stat)
        if (.not. stat%is_ok()) return
        call check_tensor_shape(self, layer%attn_out, 2, self%attention_heads * self%value_length, &
            self%hidden_size, stat)
        if (.not. stat%is_ok()) return
        call check_tensor_shape(self, layer%q_norm, 1, self%attention_head_size, 0, stat)
        if (.not. stat%is_ok()) return
        call check_tensor_shape(self, layer%k_norm, 1, self%attention_head_size, 0, stat)
        if (.not. stat%is_ok()) return
        call check_tensor_shape(self, self%mtp_eh_proj, 2, 2 * self%hidden_size, self%hidden_size, stat)
        if (.not. stat%is_ok()) return
        call check_tensor_shape(self, self%mtp_enorm, 1, self%hidden_size, 0, stat)
        if (.not. stat%is_ok()) return
        call check_tensor_shape(self, self%mtp_hnorm, 1, self%hidden_size, 0, stat)
        if (.not. stat%is_ok()) return
        if (self%mtp_embed_tokens > 0) then
            call check_tensor_shape(self, self%mtp_embed_tokens, 2, self%hidden_size, &
                self%vocabulary_size, stat)
            if (.not. stat%is_ok()) return
        end if
        call check_tensor_shape(self, self%mtp_shared_head_norm, 1, self%hidden_size, 0, stat)
        if (.not. stat%is_ok()) return
        if (self%mtp_shared_head_head > 0) then
            call check_tensor_shape(self, self%mtp_shared_head_head, 2, self%hidden_size, &
                self%vocabulary_size, stat)
            if (.not. stat%is_ok()) return
        end if
        self%mtp_layer = layer
    end subroutine bind_mtp_layer

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

    logical function cuda_q8_on_second(self, tensor_index)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: tensor_index

        cuda_q8_on_second = .false.
        if (.not. self%cuda_q8_split) return
        if (.not. allocated(self%cuda_weight_device)) return
        if (tensor_index <= 0 .or. tensor_index > size(self%cuda_weight_device)) return
        cuda_q8_on_second = self%cuda_weight_device(tensor_index) == 2
    end function cuda_q8_on_second

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
        if (self%cuda_enabled .and. cuda_host_q8_enabled() .and. &
                .not. self%cuda_q8_cpu_override .and. &
                size(self%file%tensors(tensor_index)%shape) == 2) then
                if (mod(size(input), 32) /= 0) then
                    call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA Q8 input is not block aligned')
                    return
                end if
                call fortai_q8_quantize(input, self%quantized_input, self%quantized_scales, &
                    int(size(input), c_int64_t))
                call cuda_model_matvec_quantized(self, tensor_index, size(input), output, stat)
                if (.not. stat%is_ok()) then
                    call self%file%tensors(tensor_index)%matvec_q8(input, output, &
                        self%quantized_input, self%quantized_scales, stat, self%persistent_openmp_active)
                end if
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
            if (self%cuda_enabled .and. cuda_host_q8_enabled() .and. .not. self%cuda_q8_cpu_override) then
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
                        if (.not. stat%is_ok()) then
                            call self%file%tensors(first_index)%matvec_pair_q8( &
                                self%file%tensors(second_index), input, first_output, second_output, &
                                self%quantized_input, self%quantized_scales, stat, self%persistent_openmp_active)
                        end if
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
            if (self%cuda_enabled .and. cuda_host_q8_enabled() .and. .not. self%cuda_q8_cpu_override) then
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
                            if (.not. stat%is_ok()) then
                                call self%file%tensors(first_index)%matvec_triplet_q8( &
                                    self%file%tensors(second_index), self%file%tensors(third_index), input, &
                                    first_output, second_output, third_output, self%quantized_input, &
                                    self%quantized_scales, stat, self%persistent_openmp_active)
                            end if
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
        if (cuda_q8_on_second(self, tensor_index)) then
            call cuda_q8_matvec_host(self%cuda_second, self%cuda_weights(tensor_index), &
                self%quantized_input(1:activation_bytes), int(activation_bytes, c_size_t), output, &
                int(size(output) * storage_size(output(1)) / 8, c_size_t), elapsed_ms, stat)
        else
            call cuda_q8_matvec_host(self%cuda, self%cuda_weights(tensor_index), &
                self%quantized_input(1:activation_bytes), int(activation_bytes, c_size_t), &
                output, int(size(output) * storage_size(output(1)) / 8, c_size_t), elapsed_ms, stat)
        end if
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
        if (cuda_q8_on_second(self, first_index) .eqv. cuda_q8_on_second(self, second_index)) then
            if (cuda_q8_on_second(self, first_index)) then
                call cuda_q8_matvec_host_pair(self%cuda_second, self%cuda_weights(first_index), &
                    self%cuda_weights(second_index), self%quantized_input(1:activation_bytes), &
                    int(activation_bytes, c_size_t), first_output, &
                    int(size(first_output) * storage_size(first_output(1)) / 8, c_size_t), second_output, &
                    int(size(second_output) * storage_size(second_output(1)) / 8, c_size_t), elapsed_ms, stat)
            else
                call cuda_q8_matvec_host_pair(self%cuda, self%cuda_weights(first_index), &
                    self%cuda_weights(second_index), self%quantized_input(1:activation_bytes), &
                    int(activation_bytes, c_size_t), first_output, &
                    int(size(first_output) * storage_size(first_output(1)) / 8, c_size_t), second_output, &
                    int(size(second_output) * storage_size(second_output(1)) / 8, c_size_t), elapsed_ms, stat)
            end if
        else
            call cuda_model_matvec_quantized(self, first_index, input_size, first_output, stat)
            if (.not. stat%is_ok()) return
            call cuda_model_matvec_quantized(self, second_index, input_size, second_output, stat)
        end if
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
        if (size(self%qkv_download_work) >= size(first_output) + size(second_output) + size(third_output) .and. &
            cuda_q8_on_second(self, first_index) .eqv. cuda_q8_on_second(self, second_index) .and. &
            cuda_q8_on_second(self, first_index) .eqv. cuda_q8_on_second(self, third_index)) then
            if (cuda_q8_on_second(self, first_index)) then
                call cuda_q8_matvec_host_triplet_contiguous(self%cuda_second, self%cuda_weights(first_index), &
                    self%cuda_weights(second_index), self%cuda_weights(third_index), &
                    self%quantized_input(1:activation_bytes), int(activation_bytes, c_size_t), &
                    self%qkv_download_work, int((size(first_output) + size(second_output) + size(third_output)) * &
                    storage_size(first_output(1)) / 8, c_size_t), elapsed_ms, stat)
            else
                call cuda_q8_matvec_host_triplet_contiguous(self%cuda, self%cuda_weights(first_index), &
                    self%cuda_weights(second_index), self%cuda_weights(third_index), &
                    self%quantized_input(1:activation_bytes), int(activation_bytes, c_size_t), &
                    self%qkv_download_work, int((size(first_output) + size(second_output) + size(third_output)) * &
                    storage_size(first_output(1)) / 8, c_size_t), elapsed_ms, stat)
            end if
            if (stat%is_ok()) then
                first_output = self%qkv_download_work(1:size(first_output))
                second_output = self%qkv_download_work(size(first_output) + 1:size(first_output) + size(second_output))
                third_output = self%qkv_download_work(size(first_output) + size(second_output) + 1: &
                    size(first_output) + size(second_output) + size(third_output))
            end if
        else
            call cuda_model_matvec_quantized(self, first_index, input_size, first_output, stat)
            if (.not. stat%is_ok()) return
            call cuda_model_matvec_quantized(self, second_index, input_size, second_output, stat)
            if (.not. stat%is_ok()) return
            call cuda_model_matvec_quantized(self, third_index, input_size, third_output, stat)
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
        if (cuda_q8_on_second(self, layer%ffn_gate) .eqv. cuda_q8_on_second(self, layer%ffn_up) .and. &
            cuda_q8_on_second(self, layer%ffn_gate) .eqv. cuda_q8_on_second(self, layer%ffn_down)) then
            if (cuda_q8_on_second(self, layer%ffn_gate)) then
                call cuda_q8_ffn_host(self%cuda_second, self%cuda_weights(layer%ffn_gate), &
                    self%cuda_weights(layer%ffn_up), self%cuda_weights(layer%ffn_down), &
                    self%quantized_input(1:activation_bytes), int(activation_bytes, c_size_t), output, &
                    int(size(output) * storage_size(output(1)) / 8, c_size_t), elapsed_ms, stat)
            else
                call cuda_q8_ffn_host(self%cuda, self%cuda_weights(layer%ffn_gate), &
                    self%cuda_weights(layer%ffn_up), self%cuda_weights(layer%ffn_down), &
                    self%quantized_input(1:activation_bytes), int(activation_bytes, c_size_t), output, &
                    int(size(output) * storage_size(output(1)) / 8, c_size_t), elapsed_ms, stat)
            end if
        else
            call cuda_model_matvec_quantized(self, layer%ffn_gate, input_size, self%ffn_gate_work, stat)
            if (.not. stat%is_ok()) return
            call cuda_model_matvec_quantized(self, layer%ffn_up, input_size, self%ffn_up_work, stat)
            if (.not. stat%is_ok()) return
            call fortai_silu_product(self%ffn_gate_work, self%ffn_up_work, &
                int(size(self%ffn_gate_work), c_int64_t))
            call cuda_model_matvec_quantized(self, layer%ffn_down, size(self%ffn_gate_work), output, stat)
        end if
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
        integer :: i, head, offset, scale_offset, key_row, value_row, key_blocks, value_blocks

        do head = 1, self % attention_heads_kv
            offset = int(position) * self % attention_head_size * self % attention_heads_kv + &
                (head - 1) * self % attention_head_size
            if (self%cache_key_q8) then
                key_blocks = (self%attention_head_size + 31) / 32
                key_row = self%attention_head_size + 2 * key_blocks
                offset = (int(position) * self%attention_heads_kv + head - 1) * key_row
                scale_offset = (int(position) * self%attention_heads_kv + head - 1) * key_blocks
                call fortai_q8_quantize(self%k_work((head - 1) * self%attention_head_size + 1:), &
                    layer%key_cache_q8(offset + 1:), layer%key_cache_q8_scales(scale_offset + 1:), &
                    int(self%attention_head_size, c_int64_t))
            else
                do i = 1, self % attention_head_size
                    layer%key_cache(offset + i) = gguf_fp16_to_real(fortai_float_to_half( &
                        self%k_work((head - 1) * self%attention_head_size + i)))
                end do
            end if
            offset = int(position) * self % value_length * self % attention_heads_kv + &
                (head - 1) * self % value_length
            if (self%cache_value_q8) then
                value_blocks = (self%value_length + 31) / 32
                value_row = self%value_length + 2 * value_blocks
                offset = (int(position) * self%attention_heads_kv + head - 1) * value_row
                scale_offset = (int(position) * self%attention_heads_kv + head - 1) * value_blocks
                call fortai_q8_quantize(self%v_work((head - 1) * self%value_length + 1:), &
                    layer%value_cache_q8(offset + 1:), layer%value_cache_q8_scales(scale_offset + 1:), &
                    int(self%value_length, c_int64_t))
            else
                do i = 1, self % value_length
                    layer % value_cache(offset + i) = gguf_fp16_to_real(fortai_float_to_half( &
                        self % v_work((head - 1) * self % value_length + i)))
                end do
            end if
        end do
    end subroutine qwen35_cpu_model_layers_state_store

    real(real32) function qwen35_cpu_model_layers_key_value(self, layer, head, position, index, key)
        class(qwen35_cpu_model_t), intent(in) :: self
        type(qwen35_cpu_layer_t), intent(in) :: layer
        integer, intent(in) :: head, index
        integer, intent(in) :: position
        logical, intent(in) :: key
        integer :: offset, blocks, row, scale_offset, qvalue

        if (key) then
            if (self%cache_key_q8) then
                blocks = (self%attention_head_size + 31) / 32
                row = self%attention_head_size + 2 * blocks
                offset = (position * self%attention_heads_kv + head) * row + 2 + index
                scale_offset = (position * self%attention_heads_kv + head) * blocks + index / 32 + 1
                qvalue = int(layer%key_cache_q8(offset))
                qwen35_cpu_model_layers_key_value = layer%key_cache_q8_scales(scale_offset) * real(qvalue, real32)
            else
                offset = position * self % attention_head_size * self % attention_heads_kv + &
                    head * self % attention_head_size + index
                qwen35_cpu_model_layers_key_value = layer % key_cache(offset)
            end if
        else
            if (self%cache_value_q8) then
                blocks = (self%value_length + 31) / 32
                row = self%value_length + 2 * blocks
                offset = (position * self%attention_heads_kv + head) * row + 2 + index
                scale_offset = (position * self%attention_heads_kv + head) * blocks + index / 32 + 1
                qvalue = int(layer%value_cache_q8(offset))
                qwen35_cpu_model_layers_key_value = layer%value_cache_q8_scales(scale_offset) * real(qvalue, real32)
            else
                offset = position * self % value_length * self % attention_heads_kv + &
                    head * self % value_length + index
                qwen35_cpu_model_layers_key_value = layer % value_cache(offset)
            end if
        end if
    end function qwen35_cpu_model_layers_key_value

    subroutine dequantize_attention_cache(self, layer, head, position, key, output)
        class(qwen35_cpu_model_t), intent(in) :: self
        type(qwen35_cpu_layer_t), intent(in) :: layer
        integer, intent(in) :: head
        integer(int64), intent(in) :: position
        logical, intent(in) :: key
        real(real32), contiguous, intent(out) :: output(:)
        integer :: width, blocks, row, scale_offset, element_offset
        integer :: current_position, block, i

        if (key) then
            width = self%attention_head_size
            blocks = (width + 31) / 32
            row = width + 2 * blocks
        else
            width = self%value_length
            blocks = (width + 31) / 32
            row = width + 2 * blocks
        end if
        do current_position = 0, int(position)
            element_offset = (current_position * self%attention_heads_kv + head) * row + 2
            scale_offset = (current_position * self%attention_heads_kv + head) * blocks
            do block = 0, blocks - 1
                do i = 1, min(32, width - 32 * block)
                    if (key) then
                        output(current_position * width + block * 32 + i) = &
                            layer%key_cache_q8_scales(scale_offset + block + 1) * &
                            real(layer%key_cache_q8(element_offset + block * 34 + i), real32)
                    else
                        output(current_position * width + block * 32 + i) = &
                            layer%value_cache_q8_scales(scale_offset + block + 1) * &
                            real(layer%value_cache_q8(element_offset + block * 34 + i), real32)
                    end if
                end do
            end do
        end do
    end subroutine dequantize_attention_cache

end module fortai_qwen35_cpu
