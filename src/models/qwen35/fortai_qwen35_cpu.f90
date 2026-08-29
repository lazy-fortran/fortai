module fortai_qwen35_cpu
    use, intrinsic :: iso_c_binding, only: c_associated, c_char, c_float, c_int, c_int16_t, c_int32_t, c_int64_t, c_int8_t, &
        c_null_char, c_null_ptr, c_ptr, c_size_t
    use, intrinsic :: iso_fortran_env, only: error_unit, int8, int32, int64, real32, real64
    use fortai_backend_cuda, only: cuda_q8_context_t, cuda_q8_matvec_host, &
        cuda_q8_ffn_host, cuda_q8_ffn_device, cuda_q8_matvec_host_pair, &
        cuda_q8_matvec_host_triplet, cuda_q8_matvec_host_triplet_contiguous, cuda_q8_weights_t, &
        cuda_q8_matvec_device_f32, cuda_q8_matvec_device_f32_pair, cuda_q8_reserve_matvec_scratch, &
        cuda_q8_matmul_device_f32, cuda_qwen35_add_device, &
        cuda_qwen35_add_matrix_device, cuda_qwen35_copy_column_device, cuda_qwen35_copy_device, &
        cuda_qwen35_concat_device, cuda_qwen35_concat_matrix_device, &
        cuda_qwen35_embedding_device, cuda_qwen35_embedding_device_batch, cuda_qwen35_rms_norm_device, &
        cuda_qwen35_rms_norm_matrix_device, cuda_qwen35_shift_target_hidden_device
    use fortai_backend_cuda, only: cuda_qwen35_silu_product_device, cuda_qwen35_silu_product_matrix_device
    use fortai_backend_cuda, only: cuda_q4_context_t, cuda_q4_weights_t, cuda_q4_matvec_host, &
        cuda_q4_matvec_host_pair, cuda_q4_matvec_host_triplet, cuda_q4_matvec_device, &
        cuda_q4_matvec_device_swiglu, cuda_q4_matvec_device_swiglu_remote_output, &
        cuda_q4_matvec_device_swiglu_down, cuda_q4_matmul_device_swiglu_down_slot, &
        cuda_q4_matvec_device_pair, cuda_q4_matvec_device_triplet, cuda_q4_matvec_device_group_remote_output, &
        cuda_q4_matvec_device_quad, cuda_q4_matvec_device_remote_input, cuda_q4_embedding_device, &
        cuda_q4_embedding_device_batch, cuda_q4_embedding_device_batch_slot
    use fortai_backend_cuda, only: cuda_q4_matmul_device_one_slot, cuda_q4_matmul_device_pair_slot, &
        cuda_q4_matmul_device_triplet_slot, cuda_q4_matmul_device_quad_slot
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
        ! Long-context layer placement may put a complete recurrent block on
        ! the secondary GPU.  Keep an independent state object there for
        ! prompt batches; scalar decode continues to use cuda_recurrent.
        type(cuda_qwen35_recurrent_t) :: cuda_recurrent_second
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
        real(real32), allocatable :: mtp_verify_hidden(:)
        integer(int8), allocatable :: quantized_input(:)
        real(real32), allocatable :: quantized_scales(:)
        type(cuda_q8_context_t) :: cuda
        type(cuda_q8_context_t) :: cuda_second
        type(cuda_q8_weights_t), allocatable :: cuda_weights(:)
        ! Only the small Q8 projections needed by a secondary-device batch
        ! layer are duplicated.  Q4 tensors retain their GGML-owned placement.
        type(cuda_q8_weights_t), allocatable :: cuda_weights_second(:)
        integer, allocatable :: cuda_weight_device(:)
        type(cuda_q4_context_t) :: cuda_q4
        type(cuda_q4_weights_t) :: cuda_mtp_embedding_weights
        type(cuda_q4_weights_t), allocatable :: cuda_q4_weights(:)
        integer, allocatable :: cuda_q4_weight_device(:)
        ! Placement of the long-context attention state.  Q4 weights follow
        ! the requested tensor split, while the much larger KV state is
        ! assigned per layer so a full primary device does not force the
        ! remaining layers back to host execution.
        logical, allocatable :: cuda_attention_on_second_layer(:)
        type(c_ptr), allocatable :: cuda_attn_norm(:)
        type(c_ptr), allocatable :: cuda_post_norm(:)
        type(c_ptr), allocatable :: cuda_attn_norm_second(:)
        type(c_ptr), allocatable :: cuda_post_norm_second(:)
        type(c_ptr) :: cuda_x = c_null_ptr
        type(c_ptr) :: cuda_residual = c_null_ptr
        type(c_ptr) :: cuda_normalized = c_null_ptr
        type(c_ptr) :: cuda_hidden = c_null_ptr
        type(c_ptr) :: cuda_output_norm = c_null_ptr
        type(c_ptr) :: cuda_output_norm_second = c_null_ptr
        type(c_ptr) :: cuda_logits = c_null_ptr
        ! Greedy decode keeps a split Q4 output projection on its owning
        ! device, matching llama.cpp's backend argmax path.  Only the final
        ! token ID crosses the PCIe boundary; the primary-device buffer above
        ! remains the fallback for sampling/diagnostics that need raw logits.
        type(c_ptr) :: cuda_logits_second = c_null_ptr
        type(c_ptr) :: cuda_qkv_device = c_null_ptr
        type(c_ptr) :: cuda_gate_device = c_null_ptr
        type(c_ptr) :: cuda_alpha_device = c_null_ptr
        type(c_ptr) :: cuda_beta_device = c_null_ptr
        type(c_ptr) :: cuda_ffn_gate_device = c_null_ptr
        type(c_ptr) :: cuda_ffn_up_device = c_null_ptr
        type(c_ptr) :: cuda_ffn_gate_device_second = c_null_ptr
        type(c_ptr) :: cuda_ffn_up_device_second = c_null_ptr
        type(c_ptr) :: cuda_attention_q_device = c_null_ptr
        type(c_ptr) :: cuda_attention_k_device = c_null_ptr
        type(c_ptr) :: cuda_attention_v_device = c_null_ptr
        type(c_ptr) :: cuda_attention_work_device = c_null_ptr
        type(c_ptr) :: cuda_attention_q_device_second = c_null_ptr
        type(c_ptr) :: cuda_attention_k_device_second = c_null_ptr
        type(c_ptr) :: cuda_attention_v_device_second = c_null_ptr
        type(c_ptr) :: cuda_attention_work_device_second = c_null_ptr
        type(c_ptr) :: fast_handle = c_null_ptr
        ! Reusable column-major [feature,batch] workspace for true prompt
        ! batching.  It is allocated once beside the scalar decode buffers;
        ! no model weights or KV state are duplicated.
        type(c_ptr) :: cuda_batch_x = c_null_ptr
        type(c_ptr) :: cuda_batch_residual = c_null_ptr
        type(c_ptr) :: cuda_batch_normalized = c_null_ptr
        type(c_ptr) :: cuda_batch_hidden = c_null_ptr
        type(c_ptr) :: cuda_batch_qkv = c_null_ptr
        type(c_ptr) :: cuda_batch_gate = c_null_ptr
        type(c_ptr) :: cuda_batch_alpha = c_null_ptr
        type(c_ptr) :: cuda_batch_beta = c_null_ptr
        type(c_ptr) :: cuda_batch_ffn_gate = c_null_ptr
        type(c_ptr) :: cuda_batch_ffn_up = c_null_ptr
        type(c_ptr) :: cuda_batch_q = c_null_ptr
        type(c_ptr) :: cuda_batch_k = c_null_ptr
        type(c_ptr) :: cuda_batch_v = c_null_ptr
        type(c_ptr) :: cuda_batch_attention = c_null_ptr
        type(c_ptr) :: cuda_batch_x_second = c_null_ptr
        type(c_ptr) :: cuda_batch_residual_second = c_null_ptr
        type(c_ptr) :: cuda_batch_normalized_second = c_null_ptr
        type(c_ptr) :: cuda_batch_hidden_second = c_null_ptr
        type(c_ptr) :: cuda_batch_qkv_second = c_null_ptr
        type(c_ptr) :: cuda_batch_gate_second = c_null_ptr
        type(c_ptr) :: cuda_batch_alpha_second = c_null_ptr
        type(c_ptr) :: cuda_batch_beta_second = c_null_ptr
        type(c_ptr) :: cuda_batch_ffn_gate_second = c_null_ptr
        type(c_ptr) :: cuda_batch_ffn_up_second = c_null_ptr
        type(c_ptr) :: cuda_batch_q_second = c_null_ptr
        type(c_ptr) :: cuda_batch_k_second = c_null_ptr
        type(c_ptr) :: cuda_batch_v_second = c_null_ptr
        type(c_ptr) :: cuda_batch_attention_second = c_null_ptr
        type(c_ptr) :: cuda_spec_logits = c_null_ptr
        type(c_ptr) :: cuda_spec_logits_second = c_null_ptr
        type(c_ptr) :: cuda_mtp_attn_norm = c_null_ptr
        type(c_ptr) :: cuda_mtp_post_norm = c_null_ptr
        type(c_ptr) :: cuda_mtp_enorm = c_null_ptr
        type(c_ptr) :: cuda_mtp_hnorm = c_null_ptr
        type(c_ptr) :: cuda_mtp_head_norm = c_null_ptr
        type(c_ptr) :: cuda_mtp_pending_hidden = c_null_ptr
        type(c_ptr) :: cuda_mtp_verify_hidden = c_null_ptr
        real(real32), allocatable :: cuda_batch_bridge(:)
        integer :: cuda_batch_capacity = 0
        integer :: cuda_batch_capacity_second = 0
        logical :: cuda_batch_enabled = .false.
        logical :: cuda_batch_enabled_second = .false.
        character(len=:), allocatable :: model_path
        logical :: cuda_enabled = .false.
        logical :: cuda_device_pipeline = .false.
        logical :: cuda_graph_enabled = .false.
        logical :: cuda_graph_ready = .false.
        ! Split-Q4 graph execution mirrors llama.cpp's scheduler: a graph is
        ! captured only for the contiguous primary-device prefix, while any
        ! cross-device bridge remains outside capture.  The warmup flag keeps
        ! lazy GGML/CUDA scratch allocation out of the capture itself.
        logical :: cuda_segment_graph_enabled = .false.
        logical :: cuda_segment_graph_ready(2) = .false.
        logical :: cuda_segment_graph_warmup(2) = .false.
        integer :: cuda_segment_graph_end = 0
        logical :: cuda_batch_graph_ready(64) = .false.
        logical :: cuda_batch_graph_warmup(64) = .false.
        logical :: cuda_batch_graph_disabled(64) = .false.
        logical :: cuda_mtp_graph_ready = .false.
        logical :: cuda_mtp_graph_warmup = .false.
        logical :: cuda_mtp_graph_disabled = .false.
        logical :: cuda_q4_resident = .false.
        logical :: cuda_q4_split = .false.
        logical :: cuda_q8_split = .false.
        logical :: cuda_q8_cpu_override = .false.
        logical :: cuda_q4_group_enabled = .true.
        logical :: fast_enabled = .false.
        logical :: fast_gpu = .false.
        logical :: mtp_available = .false.
        logical :: mtp_active = .false.
        integer :: mtp_cuda_slot = -1
        integer(int64) :: mtp_last_pair_position = -1_int64
        integer(int64) :: mtp_last_target_position = -1_int64
        integer(int64) :: mtp_last_pair_token = -1_int64
        integer(int64) :: mtp_last_draft_token = -1_int64
        integer(int64) :: mtp_device_draft_token = -1_int64
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
        procedure :: forward_batch => qwen35_cpu_forward_batch
        procedure :: forward_batch_verify => qwen35_cpu_forward_batch_verify
        procedure :: batch_supported => qwen35_cpu_batch_supported
        procedure :: forward_greedy => qwen35_cpu_forward_greedy
        procedure :: forward_greedy_speculative => qwen35_cpu_forward_greedy_speculative
        procedure :: prepare_sampled_speculative => qwen35_cpu_prepare_sampled_speculative
        procedure :: prepare_sampled_speculative_topk => qwen35_cpu_prepare_sampled_speculative_topk
        procedure :: accept_sampled_speculative => qwen35_cpu_accept_sampled_speculative
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
        if (present(max_context)) then
            if (max_context > 0_int64) self%max_context = max_context
        end if
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
        self%mtp_device_draft_token = -1_int64
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
        if (present(max_context)) then
            if (max_context > 0_int64) then
                self % max_context = max_context
            else
                self % max_context = self % file % meta_int('qwen35.context_length', 256_int64)
            end if
        end if
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
            allocate(self%mtp_verify_hidden(3 * self%hidden_size))
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
        integer :: split_length, device_length, ios, i
        logical :: have_q4

        qwen35_cuda_second_requested = .false.
        second_device = device + 1
        if (.not. allocated(self%file%tensors)) return
        if (cuda_split_mode_none()) return
        have_q4 = .false.
        do i = 1, size(self%file%tensors)
            if (is_q4_xl_type(self%file%tensors(i)%value_type)) then
                have_q4 = .true.
                exit
            end if
        end do
        qwen35_cuda_second_requested = have_q4
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

    logical function qwen35_cuda_q8_split_requested()
        character(len=16) :: value
        integer :: length

        qwen35_cuda_q8_split_requested = .false.
        value = ''
        call get_environment_variable('FORTAI_CUDA_Q8_SPLIT', value, length=length)
        if (length <= 0) return
        select case (trim(value(:min(length, len(value)))))
        case ('1', 'true', 'on', 'yes')
            qwen35_cuda_q8_split_requested = .true.
        end select
    end function qwen35_cuda_q8_split_requested

    subroutine qwen35_cpu_enable_cuda(self, device, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: device
        type(status_t), intent(out) :: stat
        type(status_t) :: cleanup_stat
        integer :: i, j, rows, width, q4_device, q4_second_device, q8_second_device
        integer :: q4_layer_number, q4_split_layer_count
        integer(int64) :: q4_bytes(2)
        logical :: have_q4, q4_split, layer_split_mode, second_context_requested, attention_created
        logical :: tensor_split_custom
        real(real64) :: tensor_split_fraction(2), tensor_split_sum
        real(real64) :: q4_layer_position
        integer, allocatable :: q4_tensor_layer(:)
        character(len=8) :: resident_env
        character(len=8) :: group_env
        character(len=128) :: tensor_split_env
        integer :: resident_length
        integer :: group_length
        integer :: tensor_split_length, tensor_split_status
        type(c_ptr) :: q4_stream

        call stat%clear()
        call validate_cuda_split_mode(stat)
        if (.not. stat%is_ok()) return
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
        if (allocated(self%cuda_weights)) then
            call cuda_device_pipeline_cleanup(self, cleanup_stat)
            if (allocated(self%layers)) then
                do i = 1, size(self%layers)
                    call self%layers(i)%cuda_recurrent%destroy(cleanup_stat)
                    call self%layers(i)%cuda_recurrent_second%destroy(cleanup_stat)
                    call self%layers(i)%cuda_attention%destroy(cleanup_stat)
                end do
            end if
            do i = 1, size(self%cuda_weights)
                call self%cuda_weights(i)%destroy(cleanup_stat)
            end do
            deallocate(self%cuda_weights)
            if (allocated(self%cuda_weight_device)) deallocate(self%cuda_weight_device)
        end if
        if (allocated(self%cuda_weights_second)) then
            do i = 1, size(self%cuda_weights_second)
                call self%cuda_weights_second(i)%destroy(cleanup_stat)
            end do
            deallocate(self%cuda_weights_second)
        end if
        if (allocated(self%cuda_q4_weights)) then
            do i = 1, size(self%cuda_q4_weights)
                call self%cuda_q4_weights(i)%destroy(cleanup_stat)
            end do
            deallocate(self%cuda_q4_weights)
        end if
        call self%cuda_mtp_embedding_weights%destroy(cleanup_stat)
        if (allocated(self%cuda_q4_weight_device)) deallocate(self%cuda_q4_weight_device)
        if (allocated(self%cuda_attention_on_second_layer)) deallocate(self%cuda_attention_on_second_layer)
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

        second_context_requested = qwen35_cuda_second_requested(self, device, q8_second_device)
        self%cuda_q8_split = second_context_requested .and. qwen35_cuda_q8_split_requested()
        if (second_context_requested) then
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
            if (cuda_split_mode_none()) q4_split = .false.
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
            allocate(self%cuda_q4_weight_device(size(self%file%tensors)))
            self%cuda_q4_weight_device = 1
            layer_split_mode = cuda_split_mode_layer()
            q4_split_layer_count = max(1, int(self%layer_count) + 1)
            if (layer_split_mode .and. q4_split) then
                allocate(q4_tensor_layer(size(self%file%tensors)))
                q4_tensor_layer = 0
                do i = 1, size(self%file%tensors)
                    if (.not. is_q4_xl_type(self%file%tensors(i)%value_type)) cycle
                    if (size(self%file%tensors(i)%shape) /= 2) cycle
                    if (is_unused_q4_tensor(self, i)) cycle
                    q4_tensor_layer(i) = q4_layer_for_tensor(self, i)
                end do
            end if
            q4_bytes = 0_int64
            do i = 1, size(self%file%tensors)
                if (.not. is_q4_xl_type(self%file%tensors(i)%value_type)) cycle
                if (size(self%file%tensors(i)%shape) /= 2) cycle
                if (is_unused_q4_tensor(self, i)) cycle
                q4_layer_number = 0
                if (layer_split_mode .and. q4_split) q4_layer_number = q4_tensor_layer(i)
                ! The token lookup is on the critical path of every decode.
                ! Keep an untied Q4 embedding on the primary board so its
                ! get-rows kernel can feed the resident stream directly.
                if (i == self%token_embedding) then
                    q4_device = 0
                else if (q4_layer_number > 0) then
                    ! Match llama.cpp's layer split: tensor_split is a
                    ! normalized cumulative boundary over layer_count + 1
                    ! entries (the final entry is the output layer).  All
                    ! tensors belonging to a layer stay together, which is
                    ! essential for the Q/K/V grouped path on non-peer GPUs.
                    q4_layer_position = real(q4_layer_number - 1, real64) / &
                        real(q4_split_layer_count, real64)
                    q4_device = 1
                    if (q4_layer_position < tensor_split_fraction(1)) q4_device = 0
                    ! Mixed Q4/Q8 attention blocks stay on the primary side:
                    ! the native attention kernel owns the Q8 projections and
                    ! a split mixed block would otherwise require an extra
                    ! QKV transfer.  Fully-Q4 blocks below follow the layer
                    ! split, matching llama.cpp's scheduler locality.
                    if (self%max_context <= 32768_int64 .and. q4_attention_tensor_mixed(self, i)) q4_device = 0
                    ! At short context the primary-side Q8 attention path
                    ! also owns fully-Q4 attention state.  Keep its Q/K/V
                    ! projections local so every token avoids a host bridge;
                    ! long-context placement follows the requested split.
                    if (self%max_context <= 32768_int64 .and. q4_layer_number > 0 .and. &
                        q4_layer_number <= size(self%layers) .and. .not. self%layers(q4_layer_number)%recurrent) &
                        q4_device = 0
                    ! Short decode contexts have no KV pressure; keeping the
                    ! recurrent Q4 projections beside the native GDN state
                    ! removes a bridge on every recurrent layer.  At the
                    ! production context length the normal layer split is
                    ! retained to leave room for the distributed KV cache.
                    if (self%max_context <= 32768_int64 .and. q4_recurrent_projection_tensor(self, i)) q4_device = 0
                else if (layer_split_mode .and. q4_split .and. i == self%output) then
                    q4_layer_position = real(self%layer_count, real64) / &
                        real(q4_split_layer_count, real64)
                    q4_device = 1
                    if (q4_layer_position < tensor_split_fraction(1)) q4_device = 0
                else if (.not. q4_split) then
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
                self%cuda_q4_weight_device(i) = q4_device + 1
                if (.not. stat%is_ok()) then
                    write(error_unit, '(a,i0,a,i0,a,i0,a)') 'fortai-native: Q4 upload failed at tensor ', i, &
                        ' (device slot ', q4_device, ', bytes ', size(self%file%tensors(i)%bytes), '): ' // &
                        trim(self%file%tensors(i)%name)
                    do j = 1, size(self%cuda_q4_weights)
                        call self%cuda_q4_weights(j)%destroy(cleanup_stat)
                    end do
                    deallocate(self%cuda_q4_weights)
                    deallocate(self%cuda_q4_weight_device)
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
            if (allocated(q4_tensor_layer)) deallocate(q4_tensor_layer)
            call self%cuda_q4%synchronize(stat)
            if (.not. stat%is_ok()) then
                do j = 1, size(self%cuda_q4_weights)
                    call self%cuda_q4_weights(j)%destroy(cleanup_stat)
                end do
                deallocate(self%cuda_q4_weights)
                deallocate(self%cuda_q4_weight_device)
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
            if (cuda_q4_unified_graph_requested() .or. cuda_q4_segment_graph_requested()) then
                ! In unified mode native Q8 work adopts GGML's scheduler
                ! stream.  This lets the outer resident graph contain both
                ! native kernels and direct GGML quantized launches.
                q4_stream = self%cuda_q4%stream(0)
                call self%cuda%adopt_stream(q4_stream, stat)
                if (.not. stat%is_ok()) then
                    call self%cuda_q4%destroy(cleanup_stat)
                    call self%cuda%destroy(cleanup_stat)
                    return
                end if
                if (c_associated(self%cuda_second%handle)) then
                    q4_stream = self%cuda_q4%stream(1)
                    call self%cuda_second%adopt_stream(q4_stream, stat)
                    if (.not. stat%is_ok()) then
                        call self%cuda_q4%destroy(cleanup_stat)
                        call self%cuda_second%destroy(cleanup_stat)
                        call self%cuda%destroy(cleanup_stat)
                        return
                    end if
                end if
            end if
            call self%cuda_q4%set_consumer_stream(0, self%cuda%stream(), stat)
            if (.not. stat%is_ok()) then
                do j = 1, size(self%cuda_q4_weights)
                    call self%cuda_q4_weights(j)%destroy(cleanup_stat)
                end do
                deallocate(self%cuda_q4_weights)
                deallocate(self%cuda_q4_weight_device)
                do j = 1, size(self%cuda_weights)
                    call self%cuda_weights(j)%destroy(cleanup_stat)
                end do
                deallocate(self%cuda_weights)
                call self%cuda_q4%destroy(cleanup_stat)
                call self%cuda%destroy(cleanup_stat)
                return
            end if
            if (c_associated(self%cuda_second%handle)) then
                call self%cuda_q4%set_consumer_stream(1, self%cuda_second%stream(), stat)
                if (.not. stat%is_ok()) then
                    do j = 1, size(self%cuda_q4_weights)
                        call self%cuda_q4_weights(j)%destroy(cleanup_stat)
                    end do
                    deallocate(self%cuda_q4_weights)
                    deallocate(self%cuda_q4_weight_device)
                    do j = 1, size(self%cuda_weights)
                        call self%cuda_weights(j)%destroy(cleanup_stat)
                    end do
                    deallocate(self%cuda_weights)
                    call self%cuda_q4%destroy(cleanup_stat)
                    call self%cuda_second%destroy(cleanup_stat)
                    call self%cuda%destroy(cleanup_stat)
                    return
                end if
            end if
        end if
        ! Batched split-layer execution may encounter a Q8 projection beside
        ! secondary-device Q4 weights (Qwen3.8's recurrent alpha/beta tensors
        ! are the common case).  Keep a compact second copy of Q8 matrices so
        ! the batch stays device-resident; the scalar path still uses the
        ! original placement in cuda_weights.
        if (c_associated(self%cuda_second%handle)) then
            allocate(self%cuda_weights_second(size(self%file%tensors)))
            do i = 1, size(self%file%tensors)
                if (self%file%tensors(i)%value_type /= GGML_TYPE_Q8_0) cycle
                if (size(self%file%tensors(i)%shape) /= 2) cycle
                width = int(self%file%tensors(i)%shape(1))
                rows = int(self%file%tensors(i)%shape(2))
                call self%cuda_weights_second(i)%upload(self%cuda_second, self%file%tensors(i)%bytes, &
                    int(size(self%file%tensors(i)%bytes), c_size_t), rows, width, stat)
                if (.not. stat%is_ok()) then
                    do j = 1, i - 1
                        call self%cuda_weights_second(j)%destroy(cleanup_stat)
                    end do
                    deallocate(self%cuda_weights_second)
                    return
                end if
            end do
        end if
        allocate(self%cuda_attention_on_second_layer(size(self%layers)))
        self%cuda_attention_on_second_layer = .false.
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
            else
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
            end if
            if (.not. stat%is_ok()) then
                do j = 1, size(self%layers)
                    call self%layers(j)%cuda_recurrent%destroy(cleanup_stat)
                    call self%layers(j)%cuda_recurrent_second%destroy(cleanup_stat)
                    call self%layers(j)%cuda_attention%destroy(cleanup_stat)
                end do
                do j = 1, size(self%cuda_weights)
                    call self%cuda_weights(j)%destroy(cleanup_stat)
                end do
                deallocate(self%cuda_weights)
                call self%cuda%destroy(cleanup_stat)
                return
            end if
            ! A remote recurrent block can keep its batched GDN state beside
            ! its Q4 projections.  The state object needs only the F32 SSM
            ! constants; Q8 alpha/beta matrices are duplicated above.
            if (batch_recurrent_second_candidate(self, i)) then
                call self%layers(i)%cuda_recurrent_second%create_state(self%cuda_second, &
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
                    self%recurrent_head_size, self%recurrent_inner_size, self%norm_epsilon, cleanup_stat)
                if (.not. cleanup_stat%is_ok()) then
                    call self%layers(i)%cuda_recurrent_second%destroy(cleanup_stat)
                    call cleanup_stat%clear()
                end if
            end if
        end do
        do i = 1, size(self%layers)
            if (self%layers(i)%recurrent) cycle
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
                attention_created = .false.
                ! An explicit split-attention request is a placement
                ! preference, not a requirement: if the second device cannot
                ! allocate this state, retain the primary-device fallback.
                if (cuda_attention_split_requested(self, i)) then
                    if (c_associated(self%cuda_second%handle)) then
                        call self%layers(i)%cuda_attention%create_state(self%cuda_second, &
                            self%file%tensors(self%layers(i)%q_norm)%bytes, &
                            int(size(self%file%tensors(self%layers(i)%q_norm)%bytes), c_size_t), &
                            self%file%tensors(self%layers(i)%k_norm)%bytes, &
                            int(size(self%file%tensors(self%layers(i)%k_norm)%bytes), c_size_t), &
                            self%attention_heads, self%attention_heads_kv, self%attention_head_size, &
                            self%value_length, int(self%max_context), self%rope_dimension, self%rope_base, &
                            self%norm_epsilon, cleanup_stat, self%cache_key_q8, self%cache_value_q8)
                        if (cleanup_stat%is_ok()) then
                            self%cuda_attention_on_second_layer(i) = .true.
                            attention_created = .true.
                        end if
                    end if
                end if
                if (.not. attention_created) then
                    call self%layers(i)%cuda_attention%create_state(self%cuda, &
                        self%file%tensors(self%layers(i)%q_norm)%bytes, &
                        int(size(self%file%tensors(self%layers(i)%q_norm)%bytes), c_size_t), &
                        self%file%tensors(self%layers(i)%k_norm)%bytes, &
                        int(size(self%file%tensors(self%layers(i)%k_norm)%bytes), c_size_t), &
                        self%attention_heads, self%attention_heads_kv, self%attention_head_size, &
                        self%value_length, int(self%max_context), self%rope_dimension, self%rope_base, &
                        self%norm_epsilon, cleanup_stat, self%cache_key_q8, self%cache_value_q8)
                    if (cleanup_stat%is_ok()) attention_created = .true.
                end if
                ! Allocation failure is normally device-local.  Retry on the
                ! second configured GPU so the resident pipeline can keep all
                ! layers on CUDA at the production context length.
                if (.not. attention_created .and. c_associated(self%cuda_second%handle)) then
                    call self%layers(i)%cuda_attention%create_state(self%cuda_second, &
                        self%file%tensors(self%layers(i)%q_norm)%bytes, &
                        int(size(self%file%tensors(self%layers(i)%q_norm)%bytes), c_size_t), &
                        self%file%tensors(self%layers(i)%k_norm)%bytes, &
                        int(size(self%file%tensors(self%layers(i)%k_norm)%bytes), c_size_t), &
                        self%attention_heads, self%attention_heads_kv, self%attention_head_size, &
                        self%value_length, int(self%max_context), self%rope_dimension, self%rope_base, &
                        self%norm_epsilon, cleanup_stat, self%cache_key_q8, self%cache_value_q8)
                    if (cleanup_stat%is_ok()) then
                        self%cuda_attention_on_second_layer(i) = .true.
                        attention_created = .true.
                    end if
                end if
                ! Keep the existing host fallback for genuinely unsupported
                ! attention shapes or when both GPUs are out of memory.
                if (.not. attention_created) call cleanup_stat%clear()
            end if
            if (.not. cleanup_stat%is_ok()) call cleanup_stat%clear()
        end do
        self%cuda_enabled = .true.
        call setup_cuda_device_pipeline(self, cleanup_stat)
        if (.not. cleanup_stat%is_ok()) then
            write(error_unit, '(a)') 'fortai-native: CUDA device pipeline unavailable: ' // trim(cleanup_stat%message)
            call cleanup_stat%clear()
        end if
        if (self%cuda_device_pipeline .and. self%mtp_active) then
            call setup_cuda_mtp_pipeline(self, cleanup_stat)
            if (.not. cleanup_stat%is_ok()) then
                write(error_unit, '(a)') 'fortai-native: CUDA MTP pipeline unavailable: ' // &
                    trim(cleanup_stat%message)
                self%mtp_cuda_slot = -1
                call cleanup_stat%clear()
            end if
        end if
    end subroutine qwen35_cpu_enable_cuda

    subroutine setup_cuda_mtp_pipeline(self, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(status_t), intent(out) :: stat
        integer :: slot
        integer :: projections(9)
        integer(c_size_t) :: hidden_bytes
        logical :: draft_key_q8, draft_value_q8, draft_key_q4, draft_value_q4

        call stat%clear()
        call configure_mtp_cache_types(self, draft_key_q8, draft_value_q8, draft_key_q4, &
            draft_value_q4, stat)
        if (.not. stat%is_ok()) return
        self%mtp_cuda_slot = -1
        projections = [self%mtp_eh_proj, self%mtp_layer%attn_q, self%mtp_layer%attn_k, &
            self%mtp_layer%attn_v, self%mtp_layer%attn_out, self%mtp_layer%ffn_gate, &
            self%mtp_layer%ffn_up, self%mtp_layer%ffn_down, self%mtp_output]
        do slot = 0, 1
            if (.not. batch_projection_set_supported(self, projections, slot)) cycle
            if (slot == 1 .and. .not. self%cuda_batch_enabled_second) cycle
            if (slot == 0 .and. .not. self%cuda_batch_enabled) cycle
            self%mtp_cuda_slot = slot
            exit
        end do
        if (self%mtp_cuda_slot < 0) then
            call stat%set(FORTAI_UNSUPPORTED, 'MTP projections do not share a CUDA device')
            return
        end if
        if (self%mtp_embed_tokens == 0 .and. &
            is_q4_xl_type(self%file%tensors(self%token_embedding)%value_type) .and. &
            self%cuda_q4_weight_device(self%token_embedding) /= self%mtp_cuda_slot + 1) then
            call self%cuda_mtp_embedding_weights%upload(self%cuda_q4, &
                int(self%file%tensors(self%token_embedding)%value_type), &
                self%file%tensors(self%token_embedding)%bytes, &
                int(size(self%file%tensors(self%token_embedding)%bytes), c_size_t), &
                self%vocabulary_size, self%hidden_size, self%mtp_cuda_slot, stat)
            if (.not. stat%is_ok()) return
        end if
        hidden_bytes = int(self%hidden_size * storage_size(self%x(1)) / 8, c_size_t)
        if (self%mtp_cuda_slot == 0) then
            call allocate_mtp_norms(self, self%cuda, hidden_bytes, stat)
            if (.not. stat%is_ok()) return
            call self%cuda%allocate_buffer(hidden_bytes, self%cuda_mtp_pending_hidden, stat)
            if (.not. stat%is_ok()) return
            call self%cuda%upload_real(self%cuda_mtp_pending_hidden, self%mtp_pending_hidden, stat)
            if (.not. stat%is_ok()) return
            call self%cuda%allocate_buffer(3_c_size_t * hidden_bytes, self%cuda_mtp_verify_hidden, stat)
            if (.not. stat%is_ok()) return
            call self%mtp_layer%cuda_attention%create_state(self%cuda, &
                self%file%tensors(self%mtp_layer%q_norm)%bytes, &
                int(size(self%file%tensors(self%mtp_layer%q_norm)%bytes), c_size_t), &
                self%file%tensors(self%mtp_layer%k_norm)%bytes, &
                int(size(self%file%tensors(self%mtp_layer%k_norm)%bytes), c_size_t), &
                self%attention_heads, self%attention_heads_kv, self%attention_head_size, &
                self%value_length, int(self%max_context), self%rope_dimension, self%rope_base, &
                self%norm_epsilon, stat, draft_key_q8, draft_value_q8, draft_key_q4, draft_value_q4)
        else
            call allocate_mtp_norms(self, self%cuda_second, hidden_bytes, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(hidden_bytes, self%cuda_mtp_pending_hidden, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%upload_real(self%cuda_mtp_pending_hidden, self%mtp_pending_hidden, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(3_c_size_t * hidden_bytes, self%cuda_mtp_verify_hidden, stat)
            if (.not. stat%is_ok()) return
            call self%mtp_layer%cuda_attention%create_state(self%cuda_second, &
                self%file%tensors(self%mtp_layer%q_norm)%bytes, &
                int(size(self%file%tensors(self%mtp_layer%q_norm)%bytes), c_size_t), &
                self%file%tensors(self%mtp_layer%k_norm)%bytes, &
                int(size(self%file%tensors(self%mtp_layer%k_norm)%bytes), c_size_t), &
                self%attention_heads, self%attention_heads_kv, self%attention_head_size, &
                self%value_length, int(self%max_context), self%rope_dimension, self%rope_base, &
                self%norm_epsilon, stat, draft_key_q8, draft_value_q8, draft_key_q4, draft_value_q4)
        end if
        if (.not. stat%is_ok()) self%mtp_cuda_slot = -1
    end subroutine setup_cuda_mtp_pipeline

    subroutine configure_mtp_cache_types(self, key_q8, value_q8, key_q4, value_q4, stat)
        class(qwen35_cpu_model_t), intent(in) :: self
        logical, intent(out) :: key_q8, value_q8, key_q4, value_q4
        type(status_t), intent(out) :: stat
        character(len=32) :: key_type, value_type
        integer :: key_length, value_length

        call stat%clear()
        key_type = 'f16'
        value_type = 'f16'
        key_length = 0
        value_length = 0
        call get_environment_variable('FORTAI_CACHE_TYPE_K_DRAFT', key_type, length=key_length)
        call get_environment_variable('FORTAI_CACHE_TYPE_V_DRAFT', value_type, length=value_length)
        if (key_length <= 0) then
            key_type = 'f16'
            key_length = 3
        end if
        if (value_length <= 0) then
            value_type = 'f16'
            value_length = 3
        end if
        if (key_length > len(key_type) .or. value_length > len(value_type)) then
            call stat%set(FORTAI_INVALID, 'MTP KV cache type is too long')
            return
        end if
        key_q8 = trim(key_type(:key_length)) == 'q8_0'
        value_q8 = trim(value_type(:value_length)) == 'q8_0'
        key_q4 = trim(key_type(:key_length)) == 'q4_0'
        value_q4 = trim(value_type(:value_length)) == 'q4_0'
        if (.not. (key_q8 .or. key_q4 .or. trim(key_type(:key_length)) == 'f16') .or. &
            .not. (value_q8 .or. value_q4 .or. trim(value_type(:value_length)) == 'f16')) then
            call stat%set(FORTAI_UNSUPPORTED, 'native MTP cache supports only f16, q8_0, and q4_0')
            return
        end if
        if (((key_q8 .or. key_q4) .and. mod(self%attention_head_size, 32) /= 0) .or. &
            ((value_q8 .or. value_q4) .and. mod(self%value_length, 32) /= 0)) then
            call stat%set(FORTAI_UNSUPPORTED, 'native quantized MTP cache requires 32-element blocks')
        end if
    end subroutine configure_mtp_cache_types

    subroutine allocate_mtp_norms(self, context, hidden_bytes, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(cuda_q8_context_t), intent(in) :: context
        integer(c_size_t), intent(in) :: hidden_bytes
        type(status_t), intent(out) :: stat

        call context%allocate_buffer(hidden_bytes, self%cuda_mtp_attn_norm, stat)
        if (.not. stat%is_ok()) return
        call context%upload(self%cuda_mtp_attn_norm, &
            self%file%tensors(self%mtp_layer%attn_norm)%bytes, hidden_bytes, stat)
        if (.not. stat%is_ok()) return
        call context%allocate_buffer(hidden_bytes, self%cuda_mtp_post_norm, stat)
        if (.not. stat%is_ok()) return
        call context%upload(self%cuda_mtp_post_norm, &
            self%file%tensors(self%mtp_layer%post_norm)%bytes, hidden_bytes, stat)
        if (.not. stat%is_ok()) return
        call context%allocate_buffer(hidden_bytes, self%cuda_mtp_enorm, stat)
        if (.not. stat%is_ok()) return
        call context%upload(self%cuda_mtp_enorm, self%file%tensors(self%mtp_enorm)%bytes, &
            hidden_bytes, stat)
        if (.not. stat%is_ok()) return
        call context%allocate_buffer(hidden_bytes, self%cuda_mtp_hnorm, stat)
        if (.not. stat%is_ok()) return
        call context%upload(self%cuda_mtp_hnorm, self%file%tensors(self%mtp_hnorm)%bytes, &
            hidden_bytes, stat)
        if (.not. stat%is_ok()) return
        call context%allocate_buffer(hidden_bytes, self%cuda_mtp_head_norm, stat)
        if (.not. stat%is_ok()) return
        call context%upload(self%cuda_mtp_head_norm, &
            self%file%tensors(self%mtp_shared_head_norm)%bytes, hidden_bytes, stat)
    end subroutine allocate_mtp_norms

    logical function cuda_split_mode_none()
        character(len=32) :: value
        integer :: length

        cuda_split_mode_none = .false.
        value = ''
        call get_environment_variable('FORTAI_SPLIT_MODE', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMA_ARG_SPLIT_MODE', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_SPLIT_MODE', value, length=length)
        if (length <= 0) return
        if (length > len(value)) return
        cuda_split_mode_none = trim(value(:length)) == 'none'
    end function cuda_split_mode_none

    logical function cuda_split_mode_layer()
        character(len=32) :: value
        integer :: length

        ! llama.cpp defaults to layer placement when no split mode is
        ! specified.  Keep that default here as well; tensor/row remain
        ! explicit opt-ins for their corresponding placement policies.
        cuda_split_mode_layer = .true.
        value = ''
        call get_environment_variable('FORTAI_SPLIT_MODE', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMA_ARG_SPLIT_MODE', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_SPLIT_MODE', value, length=length)
        if (length <= 0 .or. length > len(value)) return
        cuda_split_mode_layer = trim(value(:length)) == 'layer'
    end function cuda_split_mode_layer

    subroutine validate_cuda_split_mode(stat)
        type(status_t), intent(out) :: stat
        character(len=32) :: value
        integer :: length

        call stat%clear()
        value = ''
        call get_environment_variable('FORTAI_SPLIT_MODE', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMA_ARG_SPLIT_MODE', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_SPLIT_MODE', value, length=length)
        if (length <= 0) return
        if (length > len(value)) then
            call stat%set(FORTAI_INVALID, 'CUDA split mode is too long')
            return
        end if
        select case (trim(value(:length)))
        case ('none', 'layer', 'row', 'tensor')
            continue
        case default
            call stat%set(FORTAI_INVALID, 'CUDA split mode must be none, layer, row, or tensor')
        end select
    end subroutine validate_cuda_split_mode

    subroutine setup_cuda_device_pipeline(self, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(status_t), intent(out) :: stat
        integer :: i, graph_length, pipeline_length, q4_pipeline_length, q4_disable_length, batch_capacity
        integer(c_size_t) :: hidden_bytes, ffn_bytes, qkv_bytes, query_bytes, key_bytes, value_bytes
        integer(c_size_t) :: core_bytes
        integer(c_size_t) :: max_matvec_elements
        character(len=8) :: graph_env
        character(len=8) :: pipeline_env
        character(len=8) :: q4_pipeline_env, q4_disable_env
        integer :: device_layer_count
        logical :: all_device_layers
        logical, allocatable :: device_layer(:)
        type(status_t) :: cleanup_stat

        call stat%clear()
        self%cuda_device_pipeline = .false.
        self%cuda_graph_enabled = .false.
        self%cuda_graph_ready = .false.
        self%cuda_segment_graph_enabled = .false.
        self%cuda_segment_graph_ready = .false.
        self%cuda_segment_graph_warmup = .false.
        self%cuda_segment_graph_end = 0
        self%cuda_batch_graph_ready = .false.
        self%cuda_batch_graph_warmup = .false.
        self%cuda_batch_graph_disabled = .false.
        self%cuda_mtp_graph_ready = .false.
        self%cuda_mtp_graph_warmup = .false.
        self%cuda_mtp_graph_disabled = .false.
        device_layer_count = 0
        all_device_layers = .true.
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
        ! Mixed-device Q4_K_XL uses GGML-CUDA on a second scheduler and an
        ! explicit host bridge on hosts without peer access.  The resident
        ! path is the production default once CUDA flash attention is active;
        ! retain an opt-out for diagnostics and constrained deployments.
        if (self%cuda_q4_split) then
            q4_pipeline_env = ''
            call get_environment_variable('FORTAI_ENABLE_CUDA_Q4_DEVICE_PIPELINE', &
                q4_pipeline_env, length=q4_pipeline_length)
            q4_disable_env = ''
            call get_environment_variable('FORTAI_DISABLE_CUDA_Q4_DEVICE_PIPELINE', &
                q4_disable_env, length=q4_disable_length)
            if (q4_disable_length > 0 .and. q4_disable_env(1:1) == '1') return
            if (q4_pipeline_length > 0 .and. q4_pipeline_env(1:1) /= '1') return
        end if
        if (.not. allocated(self%layers)) then
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA device pipeline has no layers')
            return
        end if
        if (.not. cuda_quantized_device_ready(self, self%token_embedding)) then
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA token embedding is not device resident')
            return
        end if
        allocate(device_layer(size(self%layers)))
        device_layer = .false.
        do i = 1, size(self%layers)
            if (self%layers(i)%recurrent) then
                if (.not. c_associated(self%layers(i)%cuda_recurrent%handle)) then
                    write(error_unit, '(a,i0)') 'fortai-native: CUDA recurrent pipeline layer unavailable: ', i
                    all_device_layers = .false.
                    cycle
                end if
            else
                if (.not. c_associated(self%layers(i)%cuda_attention%handle)) then
                    write(error_unit, '(a,i0)') 'fortai-native: CUDA attention pipeline layer unavailable: ', i
                    all_device_layers = .false.
                    cycle
                end if
            end if
            if (.not. cuda_ffn_device_ready(self, self%layers(i))) then
                write(error_unit, '(a,i0)') 'fortai-native: CUDA FFN pipeline layer unavailable: ', i
                all_device_layers = .false.
                cycle
            end if
            if (size(self%file%tensors(self%layers(i)%attn_norm)%bytes) /= &
                size(self%x) * storage_size(self%x(1)) / 8) then
                write(error_unit, '(a,i0)') 'fortai-native: CUDA RMS norm unavailable for layer: ', i
                all_device_layers = .false.
                cycle
            end if
            if (size(self%file%tensors(self%layers(i)%post_norm)%bytes) /= &
                size(self%x) * storage_size(self%x(1)) / 8) then
                write(error_unit, '(a,i0)') 'fortai-native: CUDA post norm unavailable for layer: ', i
                all_device_layers = .false.
                cycle
            end if
            device_layer(i) = .true.
            device_layer_count = device_layer_count + 1
        end do
        if (device_layer_count == 0) then
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA device pipeline has no resident layers')
            return
        end if
        if (.not. all_device_layers) self%cuda_graph_enabled = .false.
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
        if (c_associated(self%cuda_second%handle)) then
            allocate(self%cuda_attn_norm_second(size(self%layers)), self%cuda_post_norm_second(size(self%layers)))
            self%cuda_attn_norm_second = c_null_ptr
            self%cuda_post_norm_second = c_null_ptr
        end if
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
        if (self%cuda_q4_split .and. c_associated(self%cuda_second%handle)) then
            call self%cuda_second%allocate_buffer(query_bytes, self%cuda_attention_q_device_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(key_bytes, self%cuda_attention_k_device_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(value_bytes, self%cuda_attention_v_device_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(core_bytes, self%cuda_attention_work_device_second, stat)
            if (.not. stat%is_ok()) return
            ! Keep a second-device FFN gate/up pair for layers whose Q4
            ! projections are entirely remote.  This avoids copying two
            ! intermediate vectors to GPU0 merely to apply SiLU before the
            ! remote down projection.
            call self%cuda_second%allocate_buffer(ffn_bytes, self%cuda_ffn_gate_device_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(ffn_bytes, self%cuda_ffn_up_device_second, stat)
            if (.not. stat%is_ok()) return
        end if
        call self%cuda%allocate_buffer(hidden_bytes, self%cuda_output_norm, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%upload(self%cuda_output_norm, self%file%tensors(self%output_norm)%bytes, &
            hidden_bytes, stat)
        if (.not. stat%is_ok()) return
        if (c_associated(self%cuda_second%handle)) then
            call self%cuda_second%allocate_buffer(hidden_bytes, self%cuda_output_norm_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%upload(self%cuda_output_norm_second, self%file%tensors(self%output_norm)%bytes, &
                hidden_bytes, stat)
            if (.not. stat%is_ok()) return
        end if
        call self%cuda%allocate_buffer(int(size(self%logits) * storage_size(self%logits(1)) / 8, c_size_t), &
            self%cuda_logits, stat)
        if (.not. stat%is_ok()) return
        ! The output tensor is normally the largest remote Q4 projection.  A
        ! small remote logits buffer lets greedy argmax run where that
        ! projection already ran, avoiding a full-vocabulary device-to-device
        ! copy for every generated token.
        if (self%cuda_q4_resident .and. self%cuda_q4_split .and. &
            c_associated(self%cuda_second%handle) .and. is_q4_xl_type(self%file%tensors(self%output)%value_type) .and. &
            cuda_q4_on_second(self, self%output)) then
            call self%cuda_second%allocate_buffer(int(size(self%logits) * storage_size(self%logits(1)) / 8, c_size_t), &
                self%cuda_logits_second, stat)
            if (.not. stat%is_ok()) return
        end if
        ! Reserve the largest Q8 activation before any optional CUDA graph
        ! capture.  Graph replay cannot perform a cudaFree/cudaMalloc resize;
        ! doing this once during model setup keeps the first post-prompt
        ! scalar decode valid even when the prompt ended in a microbatch.
        max_matvec_elements = int(size(self%x), c_size_t)
        do i = 1, size(self%file%tensors)
            if (self%file%tensors(i)%value_type /= GGML_TYPE_Q8_0) cycle
            if (.not. allocated(self%file%tensors(i)%shape)) cycle
            if (size(self%file%tensors(i)%shape) /= 2) cycle
            max_matvec_elements = max(max_matvec_elements, int(self%file%tensors(i)%shape(1), c_size_t))
        end do
        call cuda_q8_reserve_matvec_scratch(self%cuda, max_matvec_elements, &
            int(size(self%logits), c_size_t), stat)
        if (.not. stat%is_ok()) return
        batch_capacity = configured_batch_capacity()
        if (batch_capacity > 1) then
            call allocate_cuda_batch_workspace(self, batch_capacity, stat)
            if (.not. stat%is_ok()) then
                ! Prompt batching is an optional acceleration.  Keep scalar
                ! CUDA decode available when a constrained card cannot spare
                ! the bounded workspace.
                write(error_unit, '(a,i0,2a)') 'FortAI batch workspace failed capacity=', batch_capacity, ': ', &
                    trim(stat%message)
                call cuda_batch_workspace_cleanup(self, cleanup_stat)
                call stat%clear()
                self%cuda_batch_capacity = 0
                self%cuda_batch_enabled = .false.
            end if
        end if
        do i = 1, size(self%layers)
            if (.not. device_layer(i)) cycle
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
            if (c_associated(self%cuda_second%handle)) then
                call self%cuda_second%allocate_buffer(hidden_bytes, self%cuda_attn_norm_second(i), stat)
                if (.not. stat%is_ok()) return
                call self%cuda_second%upload(self%cuda_attn_norm_second(i), &
                    self%file%tensors(self%layers(i)%attn_norm)%bytes, hidden_bytes, stat)
                if (.not. stat%is_ok()) return
                call self%cuda_second%allocate_buffer(hidden_bytes, self%cuda_post_norm_second(i), stat)
                if (.not. stat%is_ok()) return
                call self%cuda_second%upload(self%cuda_post_norm_second(i), &
                    self%file%tensors(self%layers(i)%post_norm)%bytes, hidden_bytes, stat)
                if (.not. stat%is_ok()) return
            end if
        end do
        ! All weight and normalization uploads are asynchronous.  Complete
        ! both Q8 streams before asking the file mapping to discard resident
        ! pages; CUDA must never be handed a page that MADV_DONTNEED can
        ! reclaim while a DMA transfer is in flight.
        call self%cuda%synchronize(stat)
        if (.not. stat%is_ok()) return
        if (c_associated(self%cuda_second%handle)) then
            call self%cuda_second%synchronize(stat)
            if (.not. stat%is_ok()) return
        end if
        self%cuda_device_pipeline = .true.
        ! A split-Q4 request still crosses the two cards through bridge
        ! streams.  CUDA graph capture cannot legally join those pre-existing
        ! bridge events (cudaErrorStreamCaptureUnjoined), even when the Q8
        ! and Q4 scheduler streams are unified.  Keep the graph opt-in for
        ! single-device Q4, but force the split path onto the deterministic
        ! stream/event schedule; the unified-stream option still removes the
        ! redundant Q8<->Q4 hand-off without risking a mid-generation abort.
        if (self%cuda_q4_resident) then
            if (self%cuda_q4_split .or. .not. cuda_q4_unified_graph_requested()) &
                self%cuda_graph_enabled = .false.
        end if
        if (self%cuda_q4_split .and. cuda_q4_segment_graph_requested()) then
            self%cuda_segment_graph_end = cuda_primary_segment_end(self)
            self%cuda_segment_graph_enabled = self%cuda_segment_graph_end > 0
            if (self%cuda_segment_graph_end > 0) then
                write(error_unit, '(a,i0)') 'fortai-native: CUDA segment graph prefix layers=', &
                    self%cuda_segment_graph_end
            end if
            ! A segment graph and GGML's per-plan graphs cannot coexist on
            ! one stream.  The Q4 backend therefore runs its graph nodes
            ! directly while this feature is active.
            if (.not. self%cuda_segment_graph_enabled) then
                self%cuda_segment_graph_end = 0
            end if
        end if
        call release_host_attention_caches(self)
        ! Device-resident base weights no longer need physical host pages.
        ! Keep the token embedding because Q4 get-rows remains on the
        ! validated host path, and keep any non-mapped MTP sidecar tensors.
        call self%file%evict_mapped_payloads(self%token_embedding)
    end subroutine setup_cuda_device_pipeline

    logical function cuda_q4_unified_graph_requested()
        character(len=16) :: value
        integer :: length

        cuda_q4_unified_graph_requested = .false.
        value = ''
        call get_environment_variable('FORTAI_ENABLE_CUDA_Q4_UNIFIED_GRAPH', value, length=length)
        if (length <= 0 .or. length > len(value)) return
        select case (trim(value(:length)))
        case ('1', 'true', 'on', 'yes')
            cuda_q4_unified_graph_requested = .true.
        end select
    end function cuda_q4_unified_graph_requested

    logical function cuda_q4_segment_graph_requested()
        character(len=16) :: value
        integer :: length

        ! Off by default.  A captured segment graph replays the launch shape and
        ! the scalar kernel arguments recorded at capture time, and the MTP
        ! draft path depends on the current position, so replaying a graph
        ! captured at one context length degrades the drafts produced at
        ! another.  Measured on Qwen3.8-27B at a 4175-token prompt: draft
        ! acceptance 64.0% with the graph against 82.3% without (llama.cpp
        ! 80.0%), and generation 40.1 against 46.4 tok/s.  Prefill is unchanged
        ! either way, so the graph currently buys nothing and costs accuracy of
        ! the drafts.  Re-enable only once the position-dependent state is read
        ! from device memory instead of being baked into the capture.
        cuda_q4_segment_graph_requested = .false.
        value = ''
        call get_environment_variable('FORTAI_ENABLE_CUDA_Q4_SEGMENT_GRAPH', value, length=length)
        if (length <= 0 .or. length > len(value)) return
        select case (trim(value(:length)))
        case ('1', 'true', 'on', 'yes')
            cuda_q4_segment_graph_requested = .true.
        case ('0', 'false', 'off', 'no')
            cuda_q4_segment_graph_requested = .false.
        end select
    end function cuda_q4_segment_graph_requested

    logical function cuda_layer_primary_device(self, layer_index)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: layer_index
        integer :: indices(20), i, tensor_index

        cuda_layer_primary_device = .false.
        if (.not. allocated(self%layers)) return
        if (layer_index <= 0 .or. layer_index > size(self%layers)) return
        if (cuda_attention_on_second(self, layer_index)) return
        indices = [self%layers(layer_index)%attn_norm, self%layers(layer_index)%post_norm, &
            self%layers(layer_index)%ffn_gate, self%layers(layer_index)%ffn_up, &
            self%layers(layer_index)%ffn_down, self%layers(layer_index)%attn_qkv, &
            self%layers(layer_index)%attn_gate, self%layers(layer_index)%attn_q, &
            self%layers(layer_index)%attn_k, self%layers(layer_index)%attn_v, &
            self%layers(layer_index)%attn_out, self%layers(layer_index)%q_norm, &
            self%layers(layer_index)%k_norm, self%layers(layer_index)%ssm_a, &
            self%layers(layer_index)%ssm_alpha, self%layers(layer_index)%ssm_beta, &
            self%layers(layer_index)%ssm_conv, self%layers(layer_index)%ssm_dt, &
            self%layers(layer_index)%ssm_norm, self%layers(layer_index)%ssm_out]
        do i = 1, size(indices)
            tensor_index = indices(i)
            if (tensor_index <= 0 .or. tensor_index > size(self%file%tensors)) cycle
            if (is_q4_xl_type(self%file%tensors(tensor_index)%value_type)) then
                if (cuda_q4_on_second(self, tensor_index)) return
            else if (self%file%tensors(tensor_index)%value_type == GGML_TYPE_Q8_0) then
                if (cuda_q8_on_second(self, tensor_index)) return
            end if
        end do
        cuda_layer_primary_device = .true.
    end function cuda_layer_primary_device

    integer function cuda_primary_segment_end(self)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer :: i

        cuda_primary_segment_end = 0
        if (.not. allocated(self%layers)) return
        do i = 1, size(self%layers)
            if (.not. cuda_layer_primary_device(self, i)) exit
            ! A segment graph may only contain layers whose complete native
            ! implementation is resident and initialized.  Keep explicit
            ! device-layer limits and diagnostic fallbacks outside capture.
            if (.not. cuda_layer_device_ready(self, i)) exit
            cuda_primary_segment_end = i
        end do
    end function cuda_primary_segment_end

    integer function configured_batch_capacity()
        character(len=32) :: value
        integer :: length, ios, parsed

        configured_batch_capacity = 256
        value = ''
        call get_environment_variable('FORTAI_UBATCH', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMA_ARG_UBATCH', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_UBATCH', value, length=length)
        if (length <= 0 .or. length > len(value)) return
        read(value(:length), *, iostat=ios) parsed
        if (ios == 0 .and. parsed >= 2) configured_batch_capacity = min(parsed, 512)
    end function configured_batch_capacity

    subroutine allocate_cuda_batch_workspace(self, capacity, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: capacity
        type(status_t), intent(out) :: stat
        integer(c_size_t) :: hidden_bytes, ffn_bytes, qkv_bytes, query_bytes, key_bytes, value_bytes, core_bytes
        integer(c_size_t) :: hidden_batch_bytes, ffn_batch_bytes, qkv_batch_bytes, query_batch_bytes
        integer(c_size_t) :: key_batch_bytes, value_batch_bytes, core_batch_bytes
        integer(c_size_t) :: spec_logits_bytes
        integer :: secondary_capacity

        call stat%clear()
        if (capacity <= 1) return
        secondary_capacity = capacity
        hidden_bytes = int(size(self%x) * storage_size(self%x(1)) / 8, c_size_t)
        ffn_bytes = int(self%feed_forward_size * storage_size(self%x(1)) / 8, c_size_t)
        qkv_bytes = int(max(self%recurrent_conv_size, 2 * self%attention_heads * self%attention_head_size) * &
            storage_size(self%x(1)) / 8, c_size_t)
        query_bytes = int(2 * self%attention_heads * self%attention_head_size * storage_size(self%x(1)) / 8, c_size_t)
        key_bytes = int(self%attention_heads_kv * self%attention_head_size * storage_size(self%x(1)) / 8, c_size_t)
        value_bytes = int(self%attention_heads_kv * self%value_length * storage_size(self%x(1)) / 8, c_size_t)
        core_bytes = int(max(self%recurrent_inner_size, self%attention_heads * self%value_length) * &
            storage_size(self%x(1)) / 8, c_size_t)
        hidden_batch_bytes = hidden_bytes * int(capacity, c_size_t)
        ffn_batch_bytes = ffn_bytes * int(capacity, c_size_t)
        qkv_batch_bytes = qkv_bytes * int(capacity, c_size_t)
        query_batch_bytes = query_bytes * int(capacity, c_size_t)
        key_batch_bytes = key_bytes * int(capacity, c_size_t)
        value_batch_bytes = value_bytes * int(capacity, c_size_t)
        core_batch_bytes = core_bytes * int(capacity, c_size_t)
        spec_logits_bytes = int(3 * self%vocabulary_size * storage_size(self%logits(1)) / 8, c_size_t)
        call self%cuda%allocate_buffer(hidden_batch_bytes, self%cuda_batch_x, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(hidden_batch_bytes, self%cuda_batch_residual, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(hidden_batch_bytes, self%cuda_batch_normalized, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(hidden_batch_bytes, self%cuda_batch_hidden, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(qkv_batch_bytes, self%cuda_batch_qkv, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(ffn_batch_bytes, self%cuda_batch_gate, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(int(self%recurrent_value_heads * capacity * &
            storage_size(self%x(1)) / 8, c_size_t), self%cuda_batch_alpha, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(int(self%recurrent_value_heads * capacity * &
            storage_size(self%x(1)) / 8, c_size_t), self%cuda_batch_beta, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(ffn_batch_bytes, self%cuda_batch_ffn_gate, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(ffn_batch_bytes, self%cuda_batch_ffn_up, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(query_batch_bytes, self%cuda_batch_q, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(key_batch_bytes, self%cuda_batch_k, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(value_batch_bytes, self%cuda_batch_v, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(core_batch_bytes, self%cuda_batch_attention, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%allocate_buffer(spec_logits_bytes, self%cuda_spec_logits, stat)
        if (.not. stat%is_ok()) return
        self%cuda_batch_capacity = capacity
        self%cuda_batch_enabled = .true.
        ! A split model keeps the hidden matrix on the device that owns the
        ! current layer block.  Allocate the same bounded scratch set on the
        ! secondary GPU; model weights and KV state are not duplicated.
        if (c_associated(self%cuda_second%handle)) then
            call self%cuda_second%allocate_buffer(hidden_batch_bytes, self%cuda_batch_x_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(hidden_batch_bytes, self%cuda_batch_residual_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(hidden_batch_bytes, self%cuda_batch_normalized_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(hidden_batch_bytes, self%cuda_batch_hidden_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(qkv_batch_bytes, self%cuda_batch_qkv_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(ffn_batch_bytes, self%cuda_batch_gate_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(int(self%recurrent_value_heads * secondary_capacity * &
                storage_size(self%x(1)) / 8, c_size_t), self%cuda_batch_alpha_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(int(self%recurrent_value_heads * secondary_capacity * &
                storage_size(self%x(1)) / 8, c_size_t), self%cuda_batch_beta_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(ffn_batch_bytes, self%cuda_batch_ffn_gate_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(ffn_batch_bytes, self%cuda_batch_ffn_up_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(query_batch_bytes, self%cuda_batch_q_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(key_batch_bytes, self%cuda_batch_k_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(value_batch_bytes, self%cuda_batch_v_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(core_batch_bytes, self%cuda_batch_attention_second, stat)
            if (.not. stat%is_ok()) return
            call self%cuda_second%allocate_buffer(spec_logits_bytes, self%cuda_spec_logits_second, stat)
            if (.not. stat%is_ok()) return
            self%cuda_batch_capacity_second = secondary_capacity
            self%cuda_batch_enabled_second = .true.
        end if
        if (.not. allocated(self%cuda_batch_bridge)) allocate(self%cuda_batch_bridge(size(self%x) * capacity))
        if (self%mtp_available) then
            if (allocated(self%mtp_concat)) deallocate(self%mtp_concat)
            allocate(self%mtp_concat(self%hidden_size * capacity))
        end if
    end subroutine allocate_cuda_batch_workspace

    subroutine cuda_batch_workspace_cleanup(self, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        type(status_t), intent(out) :: stat

        call stat%clear()
        if (c_associated(self%cuda_batch_x)) call self%cuda%free_buffer(self%cuda_batch_x, stat)
        if (c_associated(self%cuda_batch_residual)) call self%cuda%free_buffer(self%cuda_batch_residual, stat)
        if (c_associated(self%cuda_batch_normalized)) call self%cuda%free_buffer(self%cuda_batch_normalized, stat)
        if (c_associated(self%cuda_batch_hidden)) call self%cuda%free_buffer(self%cuda_batch_hidden, stat)
        if (c_associated(self%cuda_batch_qkv)) call self%cuda%free_buffer(self%cuda_batch_qkv, stat)
        if (c_associated(self%cuda_batch_gate)) call self%cuda%free_buffer(self%cuda_batch_gate, stat)
        if (c_associated(self%cuda_batch_alpha)) call self%cuda%free_buffer(self%cuda_batch_alpha, stat)
        if (c_associated(self%cuda_batch_beta)) call self%cuda%free_buffer(self%cuda_batch_beta, stat)
        if (c_associated(self%cuda_batch_ffn_gate)) call self%cuda%free_buffer(self%cuda_batch_ffn_gate, stat)
        if (c_associated(self%cuda_batch_ffn_up)) call self%cuda%free_buffer(self%cuda_batch_ffn_up, stat)
        if (c_associated(self%cuda_batch_q)) call self%cuda%free_buffer(self%cuda_batch_q, stat)
        if (c_associated(self%cuda_batch_k)) call self%cuda%free_buffer(self%cuda_batch_k, stat)
        if (c_associated(self%cuda_batch_v)) call self%cuda%free_buffer(self%cuda_batch_v, stat)
        if (c_associated(self%cuda_batch_attention)) call self%cuda%free_buffer(self%cuda_batch_attention, stat)
        if (c_associated(self%cuda_spec_logits)) call self%cuda%free_buffer(self%cuda_spec_logits, stat)
        if (c_associated(self%cuda_batch_x_second)) call self%cuda_second%free_buffer(self%cuda_batch_x_second, stat)
        if (c_associated(self%cuda_batch_residual_second)) &
            call self%cuda_second%free_buffer(self%cuda_batch_residual_second, stat)
        if (c_associated(self%cuda_batch_normalized_second)) &
            call self%cuda_second%free_buffer(self%cuda_batch_normalized_second, stat)
        if (c_associated(self%cuda_batch_hidden_second)) call self%cuda_second%free_buffer(self%cuda_batch_hidden_second, stat)
        if (c_associated(self%cuda_batch_qkv_second)) call self%cuda_second%free_buffer(self%cuda_batch_qkv_second, stat)
        if (c_associated(self%cuda_batch_gate_second)) call self%cuda_second%free_buffer(self%cuda_batch_gate_second, stat)
        if (c_associated(self%cuda_batch_alpha_second)) call self%cuda_second%free_buffer(self%cuda_batch_alpha_second, stat)
        if (c_associated(self%cuda_batch_beta_second)) call self%cuda_second%free_buffer(self%cuda_batch_beta_second, stat)
        if (c_associated(self%cuda_batch_ffn_gate_second)) &
            call self%cuda_second%free_buffer(self%cuda_batch_ffn_gate_second, stat)
        if (c_associated(self%cuda_batch_ffn_up_second)) &
            call self%cuda_second%free_buffer(self%cuda_batch_ffn_up_second, stat)
        if (c_associated(self%cuda_batch_q_second)) call self%cuda_second%free_buffer(self%cuda_batch_q_second, stat)
        if (c_associated(self%cuda_batch_k_second)) call self%cuda_second%free_buffer(self%cuda_batch_k_second, stat)
        if (c_associated(self%cuda_batch_v_second)) call self%cuda_second%free_buffer(self%cuda_batch_v_second, stat)
        if (c_associated(self%cuda_batch_attention_second)) &
            call self%cuda_second%free_buffer(self%cuda_batch_attention_second, stat)
        if (c_associated(self%cuda_spec_logits_second)) &
            call self%cuda_second%free_buffer(self%cuda_spec_logits_second, stat)
        self%cuda_batch_x = c_null_ptr
        self%cuda_batch_residual = c_null_ptr
        self%cuda_batch_normalized = c_null_ptr
        self%cuda_batch_hidden = c_null_ptr
        self%cuda_batch_qkv = c_null_ptr
        self%cuda_batch_gate = c_null_ptr
        self%cuda_batch_alpha = c_null_ptr
        self%cuda_batch_beta = c_null_ptr
        self%cuda_batch_ffn_gate = c_null_ptr
        self%cuda_batch_ffn_up = c_null_ptr
        self%cuda_batch_q = c_null_ptr
        self%cuda_batch_k = c_null_ptr
        self%cuda_batch_v = c_null_ptr
        self%cuda_batch_attention = c_null_ptr
        self%cuda_spec_logits = c_null_ptr
        self%cuda_batch_x_second = c_null_ptr
        self%cuda_batch_residual_second = c_null_ptr
        self%cuda_batch_normalized_second = c_null_ptr
        self%cuda_batch_hidden_second = c_null_ptr
        self%cuda_batch_qkv_second = c_null_ptr
        self%cuda_batch_gate_second = c_null_ptr
        self%cuda_batch_alpha_second = c_null_ptr
        self%cuda_batch_beta_second = c_null_ptr
        self%cuda_batch_ffn_gate_second = c_null_ptr
        self%cuda_batch_ffn_up_second = c_null_ptr
        self%cuda_batch_q_second = c_null_ptr
        self%cuda_batch_k_second = c_null_ptr
        self%cuda_batch_v_second = c_null_ptr
        self%cuda_batch_attention_second = c_null_ptr
        self%cuda_spec_logits_second = c_null_ptr
        self%cuda_batch_capacity = 0
        self%cuda_batch_capacity_second = 0
        self%cuda_batch_enabled = .false.
        self%cuda_batch_enabled_second = .false.
        if (allocated(self%cuda_batch_bridge)) deallocate(self%cuda_batch_bridge)
    end subroutine cuda_batch_workspace_cleanup

    subroutine release_host_attention_caches(self)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer :: i, limit_length
        character(len=16) :: limit_value

        if (.not. allocated(self%layers)) return
        ! Keep host caches available for the layer-by-layer CUDA diagnostic
        ! (the normal resident path still evicts them to save RAM).
        limit_value = ''
        call get_environment_variable('FORTAI_CUDA_DEVICE_LAYER_LIMIT', limit_value, length=limit_length)
        if (limit_length > 0) return
        do i = 1, size(self%layers)
            if (self%layers(i)%recurrent) cycle
            if (.not. cuda_layer_kernel_ready(self, i)) cycle
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
        self%cuda_segment_graph_enabled = .false.
        self%cuda_segment_graph_ready = .false.
        self%cuda_segment_graph_warmup = .false.
        self%cuda_segment_graph_end = 0
        self%cuda_batch_graph_ready = .false.
        self%cuda_batch_graph_warmup = .false.
        self%cuda_batch_graph_disabled = .false.
        self%cuda_mtp_graph_ready = .false.
        self%cuda_mtp_graph_warmup = .false.
        self%cuda_mtp_graph_disabled = .false.
        call cuda_batch_workspace_cleanup(self, stat)
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
        if (allocated(self%cuda_attn_norm_second)) then
            do i = 1, size(self%cuda_attn_norm_second)
                if (.not. c_associated(self%cuda_attn_norm_second(i))) cycle
                call self%cuda_second%free_buffer(self%cuda_attn_norm_second(i), stat)
            end do
            deallocate(self%cuda_attn_norm_second)
        end if
        if (allocated(self%cuda_post_norm_second)) then
            do i = 1, size(self%cuda_post_norm_second)
                if (.not. c_associated(self%cuda_post_norm_second(i))) cycle
                call self%cuda_second%free_buffer(self%cuda_post_norm_second(i), stat)
            end do
            deallocate(self%cuda_post_norm_second)
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
        if (c_associated(self%cuda_attention_q_device_second)) &
            call self%cuda_second%free_buffer(self%cuda_attention_q_device_second, stat)
        if (c_associated(self%cuda_attention_k_device_second)) &
            call self%cuda_second%free_buffer(self%cuda_attention_k_device_second, stat)
        if (c_associated(self%cuda_attention_v_device_second)) &
            call self%cuda_second%free_buffer(self%cuda_attention_v_device_second, stat)
        if (c_associated(self%cuda_attention_work_device_second)) &
            call self%cuda_second%free_buffer(self%cuda_attention_work_device_second, stat)
        if (c_associated(self%cuda_ffn_gate_device_second)) &
            call self%cuda_second%free_buffer(self%cuda_ffn_gate_device_second, stat)
        if (c_associated(self%cuda_ffn_up_device_second)) &
            call self%cuda_second%free_buffer(self%cuda_ffn_up_device_second, stat)
        self%cuda_attention_q_device = c_null_ptr
        self%cuda_attention_k_device = c_null_ptr
        self%cuda_attention_v_device = c_null_ptr
        self%cuda_attention_work_device = c_null_ptr
        self%cuda_attention_q_device_second = c_null_ptr
        self%cuda_attention_k_device_second = c_null_ptr
        self%cuda_attention_v_device_second = c_null_ptr
        self%cuda_attention_work_device_second = c_null_ptr
        self%cuda_ffn_gate_device_second = c_null_ptr
        self%cuda_ffn_up_device_second = c_null_ptr
        if (c_associated(self%cuda_output_norm)) call self%cuda%free_buffer(self%cuda_output_norm, stat)
        if (c_associated(self%cuda_output_norm_second)) &
            call self%cuda_second%free_buffer(self%cuda_output_norm_second, stat)
        if (c_associated(self%cuda_logits)) call self%cuda%free_buffer(self%cuda_logits, stat)
        if (c_associated(self%cuda_logits_second)) call self%cuda_second%free_buffer(self%cuda_logits_second, stat)
        if (self%mtp_cuda_slot == 0) then
            if (c_associated(self%cuda_mtp_attn_norm)) call self%cuda%free_buffer(self%cuda_mtp_attn_norm, stat)
            if (c_associated(self%cuda_mtp_post_norm)) call self%cuda%free_buffer(self%cuda_mtp_post_norm, stat)
            if (c_associated(self%cuda_mtp_enorm)) call self%cuda%free_buffer(self%cuda_mtp_enorm, stat)
            if (c_associated(self%cuda_mtp_hnorm)) call self%cuda%free_buffer(self%cuda_mtp_hnorm, stat)
            if (c_associated(self%cuda_mtp_head_norm)) call self%cuda%free_buffer(self%cuda_mtp_head_norm, stat)
            if (c_associated(self%cuda_mtp_pending_hidden)) &
                call self%cuda%free_buffer(self%cuda_mtp_pending_hidden, stat)
            if (c_associated(self%cuda_mtp_verify_hidden)) &
                call self%cuda%free_buffer(self%cuda_mtp_verify_hidden, stat)
        else if (self%mtp_cuda_slot == 1) then
            if (c_associated(self%cuda_mtp_attn_norm)) call self%cuda_second%free_buffer(self%cuda_mtp_attn_norm, stat)
            if (c_associated(self%cuda_mtp_post_norm)) call self%cuda_second%free_buffer(self%cuda_mtp_post_norm, stat)
            if (c_associated(self%cuda_mtp_enorm)) call self%cuda_second%free_buffer(self%cuda_mtp_enorm, stat)
            if (c_associated(self%cuda_mtp_hnorm)) call self%cuda_second%free_buffer(self%cuda_mtp_hnorm, stat)
            if (c_associated(self%cuda_mtp_head_norm)) call self%cuda_second%free_buffer(self%cuda_mtp_head_norm, stat)
            if (c_associated(self%cuda_mtp_pending_hidden)) &
                call self%cuda_second%free_buffer(self%cuda_mtp_pending_hidden, stat)
            if (c_associated(self%cuda_mtp_verify_hidden)) &
                call self%cuda_second%free_buffer(self%cuda_mtp_verify_hidden, stat)
        end if
        self%cuda_mtp_attn_norm = c_null_ptr
        self%cuda_mtp_post_norm = c_null_ptr
        self%cuda_mtp_enorm = c_null_ptr
        self%cuda_mtp_hnorm = c_null_ptr
        self%cuda_mtp_head_norm = c_null_ptr
        self%cuda_mtp_pending_hidden = c_null_ptr
        self%cuda_mtp_verify_hidden = c_null_ptr
        self%mtp_cuda_slot = -1
        self%cuda_logits = c_null_ptr
        self%cuda_output_norm_second = c_null_ptr
        self%cuda_logits_second = c_null_ptr
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
                    call self%layers(i)%cuda_recurrent_second%destroy(cuda_stat)
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
        if (allocated(self%cuda_weights_second)) then
            do i = 1, size(self%cuda_weights_second)
                call self%cuda_weights_second(i)%destroy(cuda_stat)
            end do
            deallocate(self%cuda_weights_second)
        end if
        if (allocated(self%cuda_weight_device)) deallocate(self%cuda_weight_device)
        if (allocated(self%cuda_q4_weights)) then
            do i = 1, size(self%cuda_q4_weights)
                call self%cuda_q4_weights(i)%destroy(cuda_stat)
            end do
            deallocate(self%cuda_q4_weights)
        end if
        call self%cuda_mtp_embedding_weights%destroy(cuda_stat)
        if (allocated(self%cuda_q4_weight_device)) deallocate(self%cuda_q4_weight_device)
        if (allocated(self%cuda_attention_on_second_layer)) deallocate(self%cuda_attention_on_second_layer)
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
        if (allocated(self%mtp_verify_hidden)) deallocate(self%mtp_verify_hidden)
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
        ! Every device call below used to discard its status.  A reset that
        ! silently fails leaves stale device state behind and the next request
        ! reports the damage from an unrelated call, so report it here.
        logical :: reset_failed

        reset_failed = .false.

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
                if (self%cuda_enabled) then
                    call self%layers(i)%cuda_recurrent%reset(cuda_stat)
                    if (.not. cuda_stat%is_ok()) reset_failed = .true.
                    call self%layers(i)%cuda_recurrent_second%reset(cuda_stat)
                    if (.not. cuda_stat%is_ok()) reset_failed = .true.
                    call self%layers(i)%cuda_attention%reset(cuda_stat)
                    if (.not. cuda_stat%is_ok()) reset_failed = .true.
                end if
            end do
        end if
        if (allocated(self%mtp_layer%key_cache)) self%mtp_layer%key_cache = 0.0_real32
        if (allocated(self%mtp_layer%value_cache)) self%mtp_layer%value_cache = 0.0_real32
        if (allocated(self%mtp_layer%key_cache_q8)) self%mtp_layer%key_cache_q8 = 0_int8
        if (allocated(self%mtp_layer%value_cache_q8)) self%mtp_layer%value_cache_q8 = 0_int8
        if (allocated(self%mtp_layer%key_cache_q8_scales)) self%mtp_layer%key_cache_q8_scales = 0.0_real32
        if (allocated(self%mtp_layer%value_cache_q8_scales)) self%mtp_layer%value_cache_q8_scales = 0.0_real32
        if (self%cuda_enabled) then
            call self%mtp_layer%cuda_attention%reset(cuda_stat)
            if (.not. cuda_stat%is_ok()) reset_failed = .true.
        end if
        if (allocated(self%mtp_pending_hidden)) self%mtp_pending_hidden = 0.0_real32
        if (c_associated(self%cuda_mtp_pending_hidden)) then
            if (self%mtp_cuda_slot == 0) then
                call self%cuda%upload_real(self%cuda_mtp_pending_hidden, self%mtp_pending_hidden, cuda_stat)
                if (.not. cuda_stat%is_ok()) reset_failed = .true.
            else if (self%mtp_cuda_slot == 1) then
                call self%cuda_second%upload_real(self%cuda_mtp_pending_hidden, self%mtp_pending_hidden, cuda_stat)
                if (.not. cuda_stat%is_ok()) reset_failed = .true.
            end if
        end if
        self%mtp_last_pair_position = -1_int64
        self%mtp_last_target_position = -1_int64
        self%mtp_last_pair_token = -1_int64
        self%mtp_last_draft_token = -1_int64
        self%mtp_last_draft_match = .false.
        if (reset_failed) write(error_unit, '(a)') &
            'fortai-native: CUDA state reset failed: ' // trim(cuda_stat%message)
    end subroutine qwen35_cpu_reset

    subroutine qwen35_cpu_forward(self, token_id, position, logits, stat, download_logits, use_mtp, save_mtp_hidden, &
            compute_logits)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id, position
        real(real32), contiguous, intent(out) :: logits(:)
        type(status_t), intent(out) :: stat
        logical, intent(in), optional :: download_logits
        logical, intent(in), optional :: use_mtp
        logical, intent(in), optional :: save_mtp_hidden
        logical, intent(in), optional :: compute_logits
        logical :: should_download, should_use_mtp, should_save_mtp, should_compute_logits

        should_download = .true.
        if (present(download_logits)) should_download = download_logits
        should_use_mtp = .true.
        if (present(use_mtp)) should_use_mtp = use_mtp
        should_save_mtp = should_use_mtp
        if (present(save_mtp_hidden)) should_save_mtp = save_mtp_hidden
        should_compute_logits = .true.
        if (present(compute_logits)) should_compute_logits = compute_logits

        if (self%fast_enabled) then
            call qwen35_cpu_forward_body(self, token_id, position, logits, stat, should_download, should_use_mtp, &
                should_save_mtp, should_compute_logits)
        else if (self%persistent_openmp .and. .not. self%cuda_device_pipeline) then
            self%persistent_openmp_active = .true.
            !$omp parallel default(none) shared(self, token_id, position, logits, stat, should_download, &
            !$omp& should_use_mtp, should_save_mtp, should_compute_logits)
            !$omp single
            call qwen35_cpu_forward_body(self, token_id, position, logits, stat, should_download, should_use_mtp, &
                should_save_mtp, should_compute_logits)
            !$omp end single
            !$omp end parallel
            self%persistent_openmp_active = .false.
        else
            call qwen35_cpu_forward_body(self, token_id, position, logits, stat, should_download, should_use_mtp, &
                should_save_mtp, should_compute_logits)
        end if
    end subroutine qwen35_cpu_forward

    logical function qwen35_cpu_batch_supported(self, count)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: count
        integer :: i

        qwen35_cpu_batch_supported = .false.
        if (count <= 0 .or. count > self%cuda_batch_capacity) return
        ! The linked GGML-CUDA IQ4_NL MMQ specialization aborts for partial
        ! 32-column tiles on this Blackwell GPU.  Keep 1--8 token speculative
        ! verification on its MMV path and expose only complete MMQ tiles for
        ! prompt batching; the server sends any short tail through the scalar
        ! oracle instead of entering a process-fatal upstream kernel.
        if (self%cuda_q4_resident .and. count > 8 .and. mod(count, 32) /= 0) return
        if (.not. self%cuda_batch_enabled .or. .not. self%cuda_device_pipeline) then
            return
        end if
        if (.not. c_associated(self%cuda_batch_x)) return
        if (.not. c_associated(self%cuda_batch_attention)) return
        if (.not. allocated(self%layers)) return
        if (.not. batch_tensor_ready_on_slot(self, self%token_embedding, 0)) return
        do i = 1, self%layer_count
            if (.not. cuda_layer_device_ready(self, i)) return
            ! A layer's complete matrix path must be owned by one GPU.  The
            ! caller moves the hidden matrix once at contiguous placement
            ! boundaries; mixed quantizer formats within a layer are fine.
            if (batch_layer_device(self, i) < 0) return
        end do
        qwen35_cpu_batch_supported = .true.
    end function qwen35_cpu_batch_supported

    integer function batch_layer_device(self, layer_index)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: layer_index
        integer :: slot, index_count
        integer :: indices(8)

        batch_layer_device = -1
        if (layer_index <= 0 .or. layer_index > size(self%layers)) return
        if (self%layers(layer_index)%recurrent) then
            index_count = 8
            indices = [self%layers(layer_index)%ffn_gate, self%layers(layer_index)%ffn_up, &
                self%layers(layer_index)%ffn_down, self%layers(layer_index)%attn_qkv, &
                self%layers(layer_index)%attn_gate, self%layers(layer_index)%ssm_alpha, &
                self%layers(layer_index)%ssm_beta, self%layers(layer_index)%ssm_out]
        else
            ! q_norm and k_norm are F32 parameters owned by the attention
            ! state, not matrix projections.  Do not include either in the
            ! quantized batch-residency test.
            index_count = 7
            indices = [self%layers(layer_index)%ffn_gate, self%layers(layer_index)%ffn_up, &
                self%layers(layer_index)%ffn_down, self%layers(layer_index)%attn_q, &
                self%layers(layer_index)%attn_k, self%layers(layer_index)%attn_v, &
                self%layers(layer_index)%attn_out, 0]
        end if
        ! Prefer the existing layer owner.  This keeps the common prefix on
        ! GPU0 and avoids an unnecessary matrix bridge when a single-device
        ! model is used.
        do slot = 0, 1
            if (slot == 1) then
                if (.not. self%cuda_batch_enabled_second .or. &
                    .not. c_associated(self%cuda_second%handle)) cycle
                if (self%layers(layer_index)%recurrent) then
                    if (.not. c_associated(self%layers(layer_index)%cuda_recurrent_second%handle)) cycle
                end if
                if (.not. self%layers(layer_index)%recurrent .and. .not. cuda_attention_on_second(self, layer_index)) cycle
            else
                if (self%layers(layer_index)%recurrent) then
                    if (.not. c_associated(self%layers(layer_index)%cuda_recurrent%handle)) cycle
                end if
                if (.not. self%layers(layer_index)%recurrent .and. cuda_attention_on_second(self, layer_index)) cycle
            end if
            if (.not. batch_projection_set_supported(self, indices(:index_count), slot)) cycle
            if (slot == 1) then
                if (.not. allocated(self%cuda_attn_norm_second) .or. &
                    .not. allocated(self%cuda_post_norm_second)) cycle
                if (layer_index > size(self%cuda_attn_norm_second) .or. &
                    layer_index > size(self%cuda_post_norm_second)) cycle
                if (.not. c_associated(self%cuda_attn_norm_second(layer_index)) .or. &
                    .not. c_associated(self%cuda_post_norm_second(layer_index))) cycle
            else
                if (.not. c_associated(self%cuda_attn_norm(layer_index)) .or. &
                    .not. c_associated(self%cuda_post_norm(layer_index))) cycle
            end if
            batch_layer_device = slot
            return
        end do
    end function batch_layer_device

    logical function batch_recurrent_second_candidate(self, layer_index)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: layer_index
        integer :: indices(5), i

        batch_recurrent_second_candidate = .false.
        if (layer_index <= 0 .or. layer_index > size(self%layers)) return
        if (.not. self%layers(layer_index)%recurrent) return
        if (.not. c_associated(self%cuda_second%handle)) return
        indices = [self%layers(layer_index)%attn_qkv, self%layers(layer_index)%attn_gate, &
            self%layers(layer_index)%ssm_alpha, self%layers(layer_index)%ssm_beta, &
            self%layers(layer_index)%ssm_out]
        batch_recurrent_second_candidate = all([(batch_tensor_ready_on_slot(self, indices(i), 1), &
            i = 1, size(indices))])
    end function batch_recurrent_second_candidate

    logical function batch_projection_set_supported(self, indices, slot)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: indices(:)
        integer, intent(in) :: slot
        integer :: i

        batch_projection_set_supported = .false.
        do i = 1, size(indices)
            if (.not. batch_tensor_ready_on_slot(self, indices(i), slot)) return
        end do
        batch_projection_set_supported = .true.
    end function batch_projection_set_supported

    logical function batch_tensor_ready_on_slot(self, tensor_index, slot)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: tensor_index, slot

        batch_tensor_ready_on_slot = .false.
        if (slot < 0 .or. slot > 1 .or. tensor_index <= 0 .or. &
            .not. allocated(self%file%tensors)) return
        if (tensor_index > size(self%file%tensors)) return
        if (self%file%tensors(tensor_index)%value_type == GGML_TYPE_Q8_0) then
            if (size(self%file%tensors(tensor_index)%shape) /= 2) return
            if (slot == 0) then
                batch_tensor_ready_on_slot = allocated(self%cuda_weights) .and. &
                    tensor_index <= size(self%cuda_weights) .and. &
                    c_associated(self%cuda_weights(tensor_index)%handle)
            else if (allocated(self%cuda_weights_second)) then
                batch_tensor_ready_on_slot = tensor_index <= size(self%cuda_weights_second) .and. &
                    c_associated(self%cuda_weights_second(tensor_index)%handle)
            end if
        else if (is_q4_xl_type(self%file%tensors(tensor_index)%value_type)) then
            if (.not. allocated(self%cuda_q4_weights) .or. tensor_index > size(self%cuda_q4_weights)) return
            if (.not. allocated(self%cuda_q4_weight_device) .or. tensor_index > size(self%cuda_q4_weight_device)) return
            if (self%cuda_q4_weight_device(tensor_index) /= slot + 1) return
            batch_tensor_ready_on_slot = c_associated(self%cuda_q4_weights(tensor_index)%handle)
        end if
    end function batch_tensor_ready_on_slot

    logical function batch_q8_tensor_ready(self, tensor_index)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: tensor_index

        batch_q8_tensor_ready = .false.
        if (tensor_index <= 0 .or. .not. allocated(self%file%tensors)) return
        if (tensor_index > size(self%file%tensors) .or. .not. allocated(self%cuda_weights)) return
        if (tensor_index > size(self%cuda_weights)) return
        batch_q8_tensor_ready = self%file%tensors(tensor_index)%value_type == GGML_TYPE_Q8_0 .and. &
            c_associated(self%cuda_weights(tensor_index)%handle)
    end function batch_q8_tensor_ready

    logical function batch_q4_tensor_ready(self, tensor_index)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: tensor_index

        batch_q4_tensor_ready = .false.
        if (tensor_index <= 0 .or. .not. allocated(self%file%tensors)) return
        if (tensor_index > size(self%file%tensors) .or. .not. self%cuda_q4_resident) return
        if (.not. allocated(self%cuda_q4_weights) .or. tensor_index > size(self%cuda_q4_weights)) return
        if (self%file%tensors(tensor_index)%value_type /= GGML_TYPE_Q3_K .and. &
            self%file%tensors(tensor_index)%value_type /= GGML_TYPE_Q4_K .and. &
            self%file%tensors(tensor_index)%value_type /= GGML_TYPE_Q5_K .and. &
            self%file%tensors(tensor_index)%value_type /= GGML_TYPE_Q6_K .and. &
            self%file%tensors(tensor_index)%value_type /= GGML_TYPE_IQ4_NL .and. &
            self%file%tensors(tensor_index)%value_type /= GGML_TYPE_IQ3_S .and. &
            self%file%tensors(tensor_index)%value_type /= GGML_TYPE_IQ4_XS) return
        if (tensor_index > size(self%cuda_q4_weight_device)) return
        ! The placement array stores one-based CUDA device slots (primary=1,
        ! secondary=2); batched work is owned by the primary stream.
        if (self%cuda_q4_weight_device(tensor_index) /= 1) return
        batch_q4_tensor_ready = c_associated(self%cuda_q4_weights(tensor_index)%handle)
    end function batch_q4_tensor_ready

    subroutine qwen35_cpu_forward_batch(self, token_ids, position_start, logits, stat, compute_logits, update_mtp, &
            verification_graphs)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), contiguous, intent(in) :: token_ids(:)
        integer(int64), intent(in) :: position_start
        real(real32), contiguous, intent(out) :: logits(:)
        type(status_t), intent(out) :: stat
        logical, intent(in), optional :: compute_logits
        logical, intent(in), optional :: update_mtp
        logical, intent(in), optional :: verification_graphs
        integer(c_int32_t), allocatable, target :: token_ids_c(:)
        integer :: batch, i, j, layer_slot, active_slot, batch_graph_index, batch_graph_segment_end
        integer :: batch_graph_slot, batch_graph_regime
        integer(c_size_t) :: hidden_elements, hidden_bytes, ffn_elements, qkv_elements
        integer(c_size_t) :: query_elements, key_elements, value_elements, core_elements, matrix_elements
        logical :: should_compute_logits, should_update_mtp, use_verification_graphs, capture_batch_segment
        type(c_ptr) :: batch_x, batch_residual, batch_normalized, batch_hidden
        type(c_ptr) :: batch_qkv, batch_gate, batch_alpha, batch_beta
        type(c_ptr) :: batch_ffn_gate, batch_ffn_up, batch_q, batch_k, batch_v, batch_attention
        type(c_ptr) :: norm_attn, norm_post, target_x

        call stat%clear()
        should_compute_logits = .true.
        if (present(compute_logits)) should_compute_logits = compute_logits
        should_update_mtp = .true.
        if (present(update_mtp)) should_update_mtp = update_mtp
        use_verification_graphs = .false.
        if (present(verification_graphs)) use_verification_graphs = verification_graphs
        batch = size(token_ids)
        if (.not. self%batch_supported(batch)) then
            call stat%set(FORTAI_UNSUPPORTED, 'Qwen3.5 CUDA prompt batch is unavailable for this layout')
            return
        end if
        if (size(logits) /= self%vocabulary_size .or. position_start < 0_int64 .or. &
            position_start + int(batch, int64) > self%max_context) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA prompt batch dimensions are invalid')
            return
        end if
        call self%cuda%set_position(int(position_start), stat)
        if (.not. stat%is_ok()) return
        if (c_associated(self%cuda_second%handle)) then
            call self%cuda_second%set_position(int(position_start), stat)
            if (.not. stat%is_ok()) return
        end if
        allocate(token_ids_c(batch))
        do i = 1, batch
            if (token_ids(i) < 0_int64 .or. token_ids(i) >= self%vocabulary_size) then
                call stat%set(FORTAI_INVALID, 'Qwen3.5 prompt batch token is outside the vocabulary')
                return
            end if
            token_ids_c(i) = int(token_ids(i), c_int32_t)
        end do
        hidden_elements = int(size(self%x), c_size_t)
        hidden_bytes = hidden_elements * int(storage_size(self%x(1)) / 8, c_size_t)
        ffn_elements = int(self%feed_forward_size, c_size_t)
        qkv_elements = int(self%recurrent_conv_size, c_size_t)
        query_elements = int(2 * self%attention_heads * self%attention_head_size, c_size_t)
        key_elements = int(self%attention_heads_kv * self%attention_head_size, c_size_t)
        value_elements = int(self%attention_heads_kv * self%value_length, c_size_t)
        core_elements = int(self%attention_heads * self%value_length, c_size_t)
        matrix_elements = hidden_elements * int(batch, c_size_t)
        active_slot = 0
        call select_batch_workspace(self, active_slot, 1, batch_x, batch_residual, batch_normalized, batch_hidden, &
            batch_qkv, batch_gate, batch_alpha, batch_beta, batch_ffn_gate, batch_ffn_up, batch_q, batch_k, batch_v, &
            batch_attention, norm_attn, norm_post)
        if (batch_q8_tensor_ready(self, self%token_embedding)) then
            call cuda_qwen35_embedding_device_batch(self%cuda, self%cuda_weights(self%token_embedding), token_ids_c, &
                batch, batch_x, matrix_elements, stat)
        else if (batch_q4_tensor_ready(self, self%token_embedding)) then
            call cuda_q4_embedding_device_batch(self%cuda_q4, self%cuda_q4_weights(self%token_embedding), token_ids_c, &
                batch, batch_x, matrix_elements, stat)
        else
            call stat%set(FORTAI_UNSUPPORTED, 'Qwen3.5 CUDA prompt batch embedding is not device resident')
            return
        end if
        if (.not. stat%is_ok()) return
        capture_batch_segment = .false.
        batch_graph_regime = merge(2, 1, position_start + int(batch - 1, int64) >= 4096_int64)
        batch_graph_slot = 63 + batch_graph_regime
        batch_graph_segment_end = 0
        i = 1
        do while (i <= self%layer_count)
            layer_slot = batch_layer_device(self, i)
            if (layer_slot < 0) then
                call stat%set(FORTAI_UNSUPPORTED, 'Qwen3.5 CUDA prompt batch layer placement is unavailable')
                return
            end if
            if (layer_slot /= active_slot) then
                if (layer_slot == 0) then
                    target_x = self%cuda_batch_x
                else
                    target_x = self%cuda_batch_x_second
                end if
                call transfer_batch_matrix(self, active_slot, batch, batch_x, target_x, stat)
                if (.not. stat%is_ok()) return
                active_slot = layer_slot
                call select_batch_workspace(self, active_slot, i, batch_x, batch_residual, batch_normalized, batch_hidden, &
                    batch_qkv, batch_gate, batch_alpha, batch_beta, batch_ffn_gate, batch_ffn_up, batch_q, batch_k, &
                    batch_v, batch_attention, norm_attn, norm_post)
            end if
            ! Norm weights are layer-specific even when consecutive layers
            ! stay on the same GPU.
            call select_batch_workspace(self, active_slot, i, batch_x, batch_residual, batch_normalized, batch_hidden, &
                batch_qkv, batch_gate, batch_alpha, batch_beta, batch_ffn_gate, batch_ffn_up, batch_q, batch_k, &
                batch_v, batch_attention, norm_attn, norm_post)
            if (use_verification_graphs .and. self%cuda_segment_graph_enabled .and. &
                .not. self%cuda_batch_graph_disabled(active_slot + 1) .and. &
                (i == 1 .or. batch_layer_device(self, i - 1) /= active_slot)) then
                batch_graph_index = active_slot + 1 + 2 * (batch_graph_regime - 1)
                batch_graph_segment_end = i
                do j = i + 1, self%layer_count
                    if (batch_layer_device(self, j) /= active_slot) exit
                    batch_graph_segment_end = j
                end do
                if (self%cuda_batch_graph_ready(batch_graph_index)) then
                    if (active_slot == 0) then
                        call self%cuda%graph_launch_slot(batch_graph_slot, stat)
                    else
                        call self%cuda_second%graph_launch_slot(batch_graph_slot, stat)
                    end if
                    if (.not. stat%is_ok()) return
                    i = batch_graph_segment_end + 1
                    cycle
                else if (self%cuda_batch_graph_warmup(batch_graph_index)) then
                    if (active_slot == 0) then
                        call self%cuda%capture_begin_slot(batch_graph_slot, stat)
                    else
                        call self%cuda_second%capture_begin_slot(batch_graph_slot, stat)
                    end if
                    if (stat%is_ok()) then
                        capture_batch_segment = .true.
                    else
                        call stat%clear()
                        self%cuda_batch_graph_disabled(batch_graph_index) = .true.
                    end if
                else
                    self%cuda_batch_graph_warmup(batch_graph_index) = .true.
                end if
            end if
            call batch_copy(self, active_slot, batch_x, batch_residual, hidden_bytes * int(batch, c_size_t), stat)
            if (.not. stat%is_ok()) return
            call batch_rms_norm(self, active_slot, batch_x, norm_attn, batch_normalized, hidden_elements, batch, &
                self%norm_epsilon, stat)
            if (.not. stat%is_ok()) return
            if (self%layers(i)%recurrent) then
                call batch_project_quad(self, active_slot, self%layers(i)%attn_qkv, self%layers(i)%attn_gate, &
                    self%layers(i)%ssm_alpha, self%layers(i)%ssm_beta, batch_normalized, hidden_elements, batch, &
                    batch_qkv, qkv_elements, batch_gate, int(self%recurrent_inner_size, c_size_t), batch_alpha, &
                    int(self%recurrent_value_heads, c_size_t), batch_beta, &
                    int(self%recurrent_value_heads, c_size_t), stat)
                if (.not. stat%is_ok()) return
                if (active_slot == 0) then
                    call self%layers(i)%cuda_recurrent%run_core_device_batch(batch_qkv, &
                        qkv_elements * int(batch, c_size_t), batch_gate, &
                        int(self%recurrent_inner_size, c_size_t) * int(batch, c_size_t), batch_alpha, &
                        int(self%recurrent_value_heads, c_size_t) * int(batch, c_size_t), batch_beta, &
                        int(self%recurrent_value_heads, c_size_t) * int(batch, c_size_t), batch, batch_attention, &
                        int(self%recurrent_inner_size, c_size_t) * int(batch, c_size_t), stat)
                else
                    call self%layers(i)%cuda_recurrent_second%run_core_device_batch(batch_qkv, &
                        qkv_elements * int(batch, c_size_t), batch_gate, &
                        int(self%recurrent_inner_size, c_size_t) * int(batch, c_size_t), batch_alpha, &
                        int(self%recurrent_value_heads, c_size_t) * int(batch, c_size_t), batch_beta, &
                        int(self%recurrent_value_heads, c_size_t) * int(batch, c_size_t), batch, batch_attention, &
                        int(self%recurrent_inner_size, c_size_t) * int(batch, c_size_t), stat)
                end if
                if (.not. stat%is_ok()) return
                call batch_project_one(self, active_slot, self%layers(i)%ssm_out, batch_attention, &
                    int(self%recurrent_inner_size, c_size_t), batch, batch_hidden, hidden_elements, stat)
            else
                call batch_project_triplet(self, active_slot, self%layers(i)%attn_q, self%layers(i)%attn_k, &
                    self%layers(i)%attn_v, batch_normalized, hidden_elements, batch, batch_q, query_elements, &
                    batch_k, key_elements, batch_v, value_elements, stat)
                if (.not. stat%is_ok()) return
                if (active_slot == 0) then
                    call self%layers(i)%cuda_attention%run_core_device_batch(batch_q, &
                        query_elements * int(batch, c_size_t), batch_k, key_elements * int(batch, c_size_t), &
                        batch_v, value_elements * int(batch, c_size_t), int(position_start), batch, batch_attention, &
                        core_elements * int(batch, c_size_t), stat)
                else
                    call self%layers(i)%cuda_attention%run_core_device_batch(batch_q, &
                        query_elements * int(batch, c_size_t), batch_k, key_elements * int(batch, c_size_t), &
                        batch_v, value_elements * int(batch, c_size_t), int(position_start), batch, batch_attention, &
                        core_elements * int(batch, c_size_t), stat)
                end if
                if (.not. stat%is_ok()) return
                call batch_project_one(self, active_slot, self%layers(i)%attn_out, batch_attention, core_elements, batch, &
                    batch_hidden, hidden_elements, stat)
            end if
            if (.not. stat%is_ok()) return
            call batch_add(self, active_slot, batch_hidden, batch_residual, batch_x, matrix_elements, stat)
            if (.not. stat%is_ok()) return
            call batch_copy(self, active_slot, batch_x, batch_residual, hidden_bytes * int(batch, c_size_t), stat)
            if (.not. stat%is_ok()) return
            call batch_rms_norm(self, active_slot, batch_x, norm_post, batch_normalized, hidden_elements, batch, &
                self%norm_epsilon, stat)
            if (.not. stat%is_ok()) return
            call batch_ffn(self, active_slot, self%layers(i)%ffn_gate, self%layers(i)%ffn_up, &
                self%layers(i)%ffn_down, batch_normalized, hidden_elements, batch, batch_hidden, hidden_elements, stat)
            if (.not. stat%is_ok()) return
            call batch_add(self, active_slot, batch_hidden, batch_residual, batch_x, matrix_elements, stat)
            if (.not. stat%is_ok()) return
            if (capture_batch_segment .and. i == batch_graph_segment_end) then
                if (active_slot == 0) then
                    call self%cuda%capture_end_slot(batch_graph_slot, stat)
                else
                    call self%cuda_second%capture_end_slot(batch_graph_slot, stat)
                end if
                if (.not. stat%is_ok()) return
                batch_graph_index = active_slot + 1 + 2 * (batch_graph_regime - 1)
                self%cuda_batch_graph_ready(batch_graph_index) = .true.
                if (active_slot == 0) then
                    call self%cuda%graph_launch_slot(batch_graph_slot, stat)
                else
                    call self%cuda_second%graph_launch_slot(batch_graph_slot, stat)
                end if
                if (.not. stat%is_ok()) return
                capture_batch_segment = .false.
            end if
            i = i + 1
        end do
        ! The scalar output/sampler ABI consumes one hidden column.  Copy only
        ! the final prompt token to the primary context; all layer work and
        ! recurrent/KV state above remains batched and device-resident.
        if (should_compute_logits) then
            if (active_slot == 0) then
                call cuda_qwen35_copy_column_device(self%cuda, batch_x, hidden_elements, batch - 1, self%cuda_x, &
                    hidden_elements, stat)
            else
                call cuda_qwen35_copy_column_device(self%cuda_second, batch_x, hidden_elements, batch - 1, &
                    self%cuda_batch_hidden_second, hidden_elements, stat)
                if (stat%is_ok()) call transfer_batch_matrix(self, 1, 1, self%cuda_batch_hidden_second, &
                    self%cuda_x, stat)
            end if
            if (.not. stat%is_ok()) return
        end if
        if (should_compute_logits) then
            call forward_output_device(self, stat)
            if (.not. stat%is_ok()) return
            call self%cuda%download_real(self%cuda_logits, logits, stat)
            if (.not. stat%is_ok()) return
        end if
        ! Advance the MTP sidecar for every target row while preserving the
        ! two-GPU ubatch pipeline.  Shifting target hidden rows entirely on
        ! the MTP device avoids the old per-ubatch download/synchronization,
        ! so GPU0 can begin the next target chunk while GPU1 catches up MTP.
        if (self%mtp_active .and. should_update_mtp) then
            if (.not. c_associated(self%cuda_mtp_pending_hidden)) then
                call stat%set(FORTAI_INVALID, 'Qwen3.5 MTP pending-hidden device state is unavailable')
                return
            end if
            ! Qwen3.5 trains NextN against h_nextn, the main model's
            ! output-normalized hidden state.  Feeding the raw post-FFN
            ! residual here changes draft logits and lowers acceptance even
            ! though target generation remains correct.
            if (active_slot == 0) then
                norm_attn = self%cuda_output_norm
            else
                norm_attn = self%cuda_output_norm_second
            end if
            call batch_rms_norm(self, active_slot, batch_x, norm_attn, batch_normalized, &
                hidden_elements, batch, self%norm_epsilon, stat)
            if (.not. stat%is_ok()) return
            target_x = batch_normalized
            if (active_slot /= self%mtp_cuda_slot) then
                call select_batch_workspace(self, self%mtp_cuda_slot, self%layer_count, batch_x, &
                    batch_residual, batch_normalized, batch_hidden, batch_qkv, batch_gate, &
                    batch_alpha, batch_beta, batch_ffn_gate, batch_ffn_up, batch_q, batch_k, &
                    batch_v, batch_attention, norm_attn, norm_post)
                call transfer_batch_matrix(self, active_slot, batch, target_x, batch_x, stat)
                if (.not. stat%is_ok()) return
                active_slot = self%mtp_cuda_slot
                target_x = batch_x
            end if
            if (active_slot == 0) then
                call cuda_qwen35_shift_target_hidden_device(self%cuda, target_x, &
                    self%cuda_mtp_pending_hidden, int(self%hidden_size, c_size_t), batch, &
                    batch_residual, stat)
            else
                call cuda_qwen35_shift_target_hidden_device(self%cuda_second, target_x, &
                    self%cuda_mtp_pending_hidden, int(self%hidden_size, c_size_t), batch, &
                    batch_residual, stat)
            end if
            if (.not. stat%is_ok()) return
            call qwen35_cpu_mtp_catchup_cuda(self, token_ids, position_start, stat=stat, &
                target_hidden_device=batch_residual)
            if (.not. stat%is_ok()) return
            self%mtp_last_target_position = position_start + int(batch, int64) - 1_int64
            if (should_compute_logits) then
                if (active_slot == 0) then
                    call self%cuda%download_real(self%cuda_mtp_pending_hidden, &
                        self%mtp_pending_hidden, stat)
                else
                    call self%cuda_second%download_real(self%cuda_mtp_pending_hidden, &
                        self%mtp_pending_hidden, stat)
                end if
                if (.not. stat%is_ok()) return
            end if
        end if
    end subroutine qwen35_cpu_forward_batch

    subroutine qwen35_cpu_forward_batch_verify(self, token_ids, position_start, greedy_tokens, stat, verify_logits, &
            verify_topk_indices, verify_topk_values)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), contiguous, intent(in) :: token_ids(:)
        integer(int64), contiguous, intent(out) :: greedy_tokens(:)
        integer(int64), intent(in) :: position_start
        type(status_t), intent(out) :: stat
        real(real32), contiguous, intent(out), optional :: verify_logits(:)
        integer(c_int), contiguous, intent(out), optional :: verify_topk_indices(:, :)
        real(c_float), contiguous, intent(out), optional :: verify_topk_values(:, :)
        integer(c_int) :: indices(3)
        integer :: active_slot, output_slot, batch
        integer(c_size_t) :: hidden_elements
        logical :: sampled_verification
        type(c_ptr) :: batch_x, batch_residual, batch_normalized, batch_hidden
        type(c_ptr) :: batch_qkv, batch_gate, batch_alpha, batch_beta
        type(c_ptr) :: batch_ffn_gate, batch_ffn_up, batch_q, batch_k, batch_v
        type(c_ptr) :: batch_attention, norm_attn, norm_post, target_x, spec_logits

        call stat%clear()
        batch = size(token_ids)
        if (batch <= 0 .or. batch > 3 .or. size(greedy_tokens) < batch .or. &
            .not. allocated(self%mtp_verify_hidden)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 verification batch must contain one to three tokens')
            return
        end if
        if (present(verify_logits)) then
            if (size(verify_logits) < batch * self%vocabulary_size) then
                call stat%set(FORTAI_INVALID, 'Qwen3.5 verification logits workspace is too small')
                return
            end if
        end if
        if (present(verify_topk_indices) .neqv. present(verify_topk_values)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 verification top-k buffers must be paired')
            return
        end if
        if (present(verify_topk_indices)) then
            if (size(verify_topk_indices, 1) <= 0 .or. size(verify_topk_indices, 1) > 32 .or. &
                size(verify_topk_indices, 2) < batch .or. &
                any(shape(verify_topk_values) /= shape(verify_topk_indices))) then
                call stat%set(FORTAI_INVALID, 'Qwen3.5 verification top-k workspace is invalid')
                return
            end if
        end if
        if (present(verify_logits) .and. present(verify_topk_indices)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 verification output mode is ambiguous')
            return
        end if
        ! Native MTP keeps the accepted target hidden row on its CUDA device
        ! for both sampled and greedy verification.  The old greedy-only path
        ! downloaded all three hidden rows here and uploaded the accepted row
        ! again during MTP catch-up, adding a stream synchronization to every
        ! speculative round.
        sampled_verification = present(verify_logits) .or. present(verify_topk_indices) .or. &
            (self%mtp_cuda_slot >= 0 .and. c_associated(self%cuda_mtp_verify_hidden))
        call self%forward_batch(token_ids, position_start, self%logits, stat, .false., .false., .true.)
        if (.not. stat%is_ok()) return

        hidden_elements = int(self%hidden_size, c_size_t)
        active_slot = batch_layer_device(self, self%layer_count)
        call select_batch_workspace(self, active_slot, self%layer_count, batch_x, batch_residual, &
            batch_normalized, batch_hidden, batch_qkv, batch_gate, batch_alpha, batch_beta, &
            batch_ffn_gate, batch_ffn_up, batch_q, batch_k, batch_v, batch_attention, norm_attn, norm_post)
        output_slot = active_slot
        if (.not. batch_tensor_ready_on_slot(self, self%output, output_slot)) output_slot = 1 - active_slot
        if (.not. batch_tensor_ready_on_slot(self, self%output, output_slot)) then
            call stat%set(FORTAI_UNSUPPORTED, 'Qwen3.5 batched output projection is unavailable')
            return
        end if
        if (output_slot /= active_slot) then
            if (output_slot == 0) then
                target_x = self%cuda_batch_x
            else
                target_x = self%cuda_batch_x_second
            end if
            call transfer_batch_matrix(self, active_slot, batch, batch_x, target_x, stat)
            if (.not. stat%is_ok()) return
            active_slot = output_slot
            call select_batch_workspace(self, active_slot, self%layer_count, batch_x, batch_residual, &
                batch_normalized, batch_hidden, batch_qkv, batch_gate, batch_alpha, batch_beta, &
                batch_ffn_gate, batch_ffn_up, batch_q, batch_k, batch_v, batch_attention, norm_attn, norm_post)
        end if

        if (active_slot == 0) then
            norm_attn = self%cuda_output_norm
            spec_logits = self%cuda_spec_logits
        else
            norm_attn = self%cuda_output_norm_second
            spec_logits = self%cuda_spec_logits_second
        end if
        call batch_rms_norm(self, active_slot, batch_x, norm_attn, batch_normalized, &
            hidden_elements, batch, self%norm_epsilon, stat)
        if (.not. stat%is_ok()) return
        if (sampled_verification) then
            if (.not. c_associated(self%cuda_mtp_verify_hidden)) then
                call stat%set(FORTAI_INVALID, 'Qwen3.5 MTP device verification state is unavailable')
                return
            end if
            if (active_slot == self%mtp_cuda_slot) then
                call batch_copy(self, active_slot, batch_normalized, self%cuda_mtp_verify_hidden, &
                    hidden_elements * int(batch, c_size_t) * int(storage_size(self%x(1)) / 8, c_size_t), stat)
            else
                call transfer_batch_matrix(self, active_slot, batch, batch_normalized, self%cuda_mtp_verify_hidden, stat)
            end if
        else if (active_slot == 0) then
            call self%cuda%download_real(batch_normalized, self%cuda_batch_bridge(:batch * self%hidden_size), stat)
            if (stat%is_ok()) self%mtp_verify_hidden(:batch * self%hidden_size) = &
                self%cuda_batch_bridge(:batch * self%hidden_size)
        else
            call self%cuda_second%download_real(batch_normalized, self%cuda_batch_bridge(:batch * self%hidden_size), stat)
            if (stat%is_ok()) self%mtp_verify_hidden(:batch * self%hidden_size) = &
                self%cuda_batch_bridge(:batch * self%hidden_size)
        end if
        if (.not. stat%is_ok()) return
        call batch_project_one(self, active_slot, self%output, batch_normalized, hidden_elements, &
            batch, spec_logits, int(self%vocabulary_size, c_size_t), stat)
        if (.not. stat%is_ok()) return
        if (active_slot == 0) then
            if (present(verify_logits)) then
                call self%cuda%download_real(spec_logits, verify_logits(:batch * self%vocabulary_size), stat)
                if (.not. stat%is_ok()) return
                greedy_tokens(:batch) = -1_int64
            else if (present(verify_topk_indices)) then
                call self%cuda%topk_rows_device(spec_logits, int(self%vocabulary_size, c_size_t), &
                    verify_topk_indices(:, :batch), verify_topk_values(:, :batch), stat)
                if (stat%is_ok()) greedy_tokens(:batch) = -1_int64
            else
                call self%cuda%argmax_rows_device(spec_logits, int(self%vocabulary_size, c_size_t), &
                    indices(:batch), stat)
                if (stat%is_ok()) greedy_tokens(:batch) = int(indices(:batch), int64)
            end if
        else
            if (present(verify_logits)) then
                call self%cuda_second%download_real(spec_logits, verify_logits(:batch * self%vocabulary_size), stat)
                if (.not. stat%is_ok()) return
                greedy_tokens(:batch) = -1_int64
            else if (present(verify_topk_indices)) then
                call self%cuda_second%topk_rows_device(spec_logits, int(self%vocabulary_size, c_size_t), &
                    verify_topk_indices(:, :batch), verify_topk_values(:, :batch), stat)
                if (stat%is_ok()) greedy_tokens(:batch) = -1_int64
            else
                call self%cuda_second%argmax_rows_device(spec_logits, int(self%vocabulary_size, c_size_t), &
                    indices(:batch), stat)
                if (stat%is_ok()) greedy_tokens(:batch) = int(indices(:batch), int64)
            end if
        end if
        if (.not. stat%is_ok()) return
    end subroutine qwen35_cpu_forward_batch_verify

    subroutine qwen35_cpu_prepare_sampled_speculative(self, token_id, position, draft_tokens, &
            verify_logits, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id, position
        integer(int64), intent(out) :: draft_tokens(:)
        real(real32), contiguous, intent(out) :: verify_logits(:)
        type(status_t), intent(out) :: stat
        integer(int64) :: verify_input(3), unused_greedy(3)
        real(real32) :: draft_sum

        call stat%clear()
        if (.not. self%mtp_active .or. native_mtp_limit() < 2 .or. &
            size(draft_tokens) < 2 .or. size(verify_logits) < 3 * self%vocabulary_size .or. &
            position + 2_int64 >= self%max_context .or. .not. self%batch_supported(3)) then
            call stat%set(FORTAI_UNSUPPORTED, 'Qwen3.5 sampled MTP depth-two verification is unavailable')
            return
        end if
        call self%mtp_draft_greedy(token_id, position, draft_tokens(1), draft_sum, stat)
        if (.not. stat%is_ok()) return
        call qwen35_cpu_mtp_continue_cuda(self, draft_tokens(1), position + 1_int64, &
            draft_tokens(2), stat)
        if (.not. stat%is_ok()) return
        verify_input = [token_id, draft_tokens(1), draft_tokens(2)]
        call self%forward_batch_verify(verify_input, position, unused_greedy, stat, verify_logits)
        if (.not. stat%is_ok()) return
        call qwen35_cpu_mtp_refresh_verify_cuda(self, verify_input, position, stat)
        if (.not. stat%is_ok()) return
        self%mtp_last_draft_token = draft_tokens(2)
    end subroutine qwen35_cpu_prepare_sampled_speculative

    subroutine qwen35_cpu_prepare_sampled_speculative_topk(self, token_id, position, draft_tokens, &
            verify_topk_indices, verify_topk_values, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id, position
        integer(int64), intent(out) :: draft_tokens(:)
        integer(c_int), contiguous, intent(out) :: verify_topk_indices(:, :)
        real(c_float), contiguous, intent(out) :: verify_topk_values(:, :)
        type(status_t), intent(out) :: stat
        integer(int64) :: verify_input(3), unused_greedy(3)
        real(real32) :: draft_sum

        call stat%clear()
        if (.not. self%mtp_active .or. native_mtp_limit() < 2 .or. size(draft_tokens) < 2) then
            call stat%set(FORTAI_UNSUPPORTED, 'Qwen3.5 sampled MTP depth two is unavailable')
            return
        end if
        if (size(verify_topk_indices, 1) <= 0 .or. size(verify_topk_indices, 1) > 32 .or. &
            size(verify_topk_indices, 2) < 3 .or. &
            any(shape(verify_topk_values) /= shape(verify_topk_indices))) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 sampled MTP top-k workspace is invalid')
            return
        end if
        if (position + 2_int64 >= self%max_context) then
            call stat%set(FORTAI_UNSUPPORTED, 'Qwen3.5 sampled MTP top-k reached the context limit')
            return
        end if
        call self%mtp_draft_greedy(token_id, position, draft_tokens(1), draft_sum, stat)
        if (.not. stat%is_ok()) return
        call qwen35_cpu_mtp_continue_cuda(self, draft_tokens(1), position + 1_int64, &
            draft_tokens(2), stat)
        if (.not. stat%is_ok()) return
        verify_input = [token_id, draft_tokens(1), draft_tokens(2)]
        call self%forward_batch_verify(verify_input, position, unused_greedy, stat, &
            verify_topk_indices=verify_topk_indices, verify_topk_values=verify_topk_values)
        if (.not. stat%is_ok()) return
        call qwen35_cpu_mtp_refresh_verify_cuda(self, verify_input, position, stat)
        if (.not. stat%is_ok()) return
        self%mtp_last_draft_token = draft_tokens(2)
    end subroutine qwen35_cpu_prepare_sampled_speculative_topk

    subroutine qwen35_cpu_accept_sampled_speculative(self, draft_tokens, position, accepted, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: draft_tokens(:), position
        integer, intent(in) :: accepted
        type(status_t), intent(out) :: stat
        integer :: i, layer_slot

        call stat%clear()
        if (size(draft_tokens) < 2 .or. accepted < 0 .or. accepted > 2 .or. &
            .not. allocated(self%mtp_verify_hidden)) then
            call stat%set(FORTAI_INVALID, 'invalid sampled MTP acceptance')
            return
        end if
        if (self%mtp_cuda_slot >= 0) then
            if (.not. c_associated(self%cuda_mtp_verify_hidden) .or. &
                .not. c_associated(self%cuda_mtp_pending_hidden)) then
                call stat%set(FORTAI_INVALID, 'invalid sampled MTP device hidden state')
                return
            end if
            if (self%mtp_cuda_slot == 0) then
                call cuda_qwen35_copy_column_device(self%cuda, self%cuda_mtp_verify_hidden, &
                    int(self%hidden_size, c_size_t), accepted, self%cuda_mtp_pending_hidden, &
                    int(self%hidden_size, c_size_t), stat)
            else
                call cuda_qwen35_copy_column_device(self%cuda_second, self%cuda_mtp_verify_hidden, &
                    int(self%hidden_size, c_size_t), accepted, self%cuda_mtp_pending_hidden, &
                    int(self%hidden_size, c_size_t), stat)
            end if
            if (.not. stat%is_ok()) return
        end if
        self%mtp_last_draft_match = accepted == 2
        if (accepted == 0) then
            do i = 1, self%layer_count
                if (.not. self%layers(i)%recurrent) cycle
                layer_slot = batch_layer_device(self, i)
                if (layer_slot == 0) then
                    call self%layers(i)%cuda_recurrent%restore_first(stat)
                else
                    call self%layers(i)%cuda_recurrent_second%restore_first(stat)
                end if
                if (.not. stat%is_ok()) return
            end do
            if (self%mtp_cuda_slot < 0) self%mtp_pending_hidden = self%mtp_verify_hidden(:self%hidden_size)
            self%mtp_last_target_position = position
        else if (accepted == 1) then
            do i = 1, self%layer_count
                if (.not. self%layers(i)%recurrent) cycle
                layer_slot = batch_layer_device(self, i)
                if (layer_slot == 0) then
                    call self%layers(i)%cuda_recurrent%restore_second(stat)
                else
                    call self%layers(i)%cuda_recurrent_second%restore_second(stat)
                end if
                if (.not. stat%is_ok()) return
            end do
            if (self%mtp_cuda_slot < 0) self%mtp_pending_hidden = self%mtp_verify_hidden( &
                self%hidden_size + 1:2 * self%hidden_size)
            self%mtp_last_target_position = position + 1_int64
            if (self%mtp_cuda_slot < 0) then
                call qwen35_cpu_mtp_catchup_cuda(self, draft_tokens(:1), position + 1_int64, &
                    self%mtp_verify_hidden(:self%hidden_size), stat)
            end if
        else
            if (self%mtp_cuda_slot < 0) self%mtp_pending_hidden = self%mtp_verify_hidden( &
                2 * self%hidden_size + 1:3 * self%hidden_size)
            self%mtp_last_target_position = position + 2_int64
            if (self%mtp_cuda_slot < 0) then
                call qwen35_cpu_mtp_catchup_cuda(self, draft_tokens(:2), position + 1_int64, &
                    self%mtp_verify_hidden(:2 * self%hidden_size), stat)
            end if
        end if
    end subroutine qwen35_cpu_accept_sampled_speculative

    subroutine select_batch_workspace(self, slot, layer_index, batch_x, batch_residual, batch_normalized, batch_hidden, &
            batch_qkv, batch_gate, batch_alpha, batch_beta, batch_ffn_gate, batch_ffn_up, batch_q, batch_k, batch_v, &
            batch_attention, norm_attn, norm_post)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: slot, layer_index
        type(c_ptr), intent(out) :: batch_x, batch_residual, batch_normalized, batch_hidden
        type(c_ptr), intent(out) :: batch_qkv, batch_gate, batch_alpha, batch_beta
        type(c_ptr), intent(out) :: batch_ffn_gate, batch_ffn_up, batch_q, batch_k, batch_v, batch_attention
        type(c_ptr), intent(out) :: norm_attn, norm_post

        if (slot == 1) then
            batch_x = self%cuda_batch_x_second
            batch_residual = self%cuda_batch_residual_second
            batch_normalized = self%cuda_batch_normalized_second
            batch_hidden = self%cuda_batch_hidden_second
            batch_qkv = self%cuda_batch_qkv_second
            batch_gate = self%cuda_batch_gate_second
            batch_alpha = self%cuda_batch_alpha_second
            batch_beta = self%cuda_batch_beta_second
            batch_ffn_gate = self%cuda_batch_ffn_gate_second
            batch_ffn_up = self%cuda_batch_ffn_up_second
            batch_q = self%cuda_batch_q_second
            batch_k = self%cuda_batch_k_second
            batch_v = self%cuda_batch_v_second
            batch_attention = self%cuda_batch_attention_second
            if (allocated(self%cuda_attn_norm_second) .and. layer_index >= 1 .and. &
                layer_index <= size(self%cuda_attn_norm_second)) then
                norm_attn = self%cuda_attn_norm_second(layer_index)
                norm_post = self%cuda_post_norm_second(layer_index)
            else
                norm_attn = c_null_ptr
                norm_post = c_null_ptr
            end if
        else
            batch_x = self%cuda_batch_x
            batch_residual = self%cuda_batch_residual
            batch_normalized = self%cuda_batch_normalized
            batch_hidden = self%cuda_batch_hidden
            batch_qkv = self%cuda_batch_qkv
            batch_gate = self%cuda_batch_gate
            batch_alpha = self%cuda_batch_alpha
            batch_beta = self%cuda_batch_beta
            batch_ffn_gate = self%cuda_batch_ffn_gate
            batch_ffn_up = self%cuda_batch_ffn_up
            batch_q = self%cuda_batch_q
            batch_k = self%cuda_batch_k
            batch_v = self%cuda_batch_v
            batch_attention = self%cuda_batch_attention
            if (allocated(self%cuda_attn_norm) .and. layer_index >= 1 .and. &
                layer_index <= size(self%cuda_attn_norm)) then
                norm_attn = self%cuda_attn_norm(layer_index)
                norm_post = self%cuda_post_norm(layer_index)
            else
                norm_attn = c_null_ptr
                norm_post = c_null_ptr
            end if
        end if
    end subroutine select_batch_workspace

    subroutine transfer_batch_matrix(self, source_slot, batch, source, destination, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: source_slot, batch
        type(c_ptr), intent(in) :: source, destination
        type(status_t), intent(out) :: stat
        integer :: elements
        integer(c_size_t) :: bytes

        call stat%clear()
        elements = size(self%x) * batch
        if (source_slot < 0 .or. source_slot > 1 .or. batch <= 0 .or. elements <= 0 .or. &
            .not. c_associated(source) .or. .not. c_associated(destination) .or. &
            .not. allocated(self%cuda_batch_bridge) .or. size(self%cuda_batch_bridge) < elements) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA batch matrix transfer')
            return
        end if
        bytes = int(elements, c_size_t) * int(storage_size(self%x(1)) / 8, c_size_t)
        call self%cuda_q4%transfer(source_slot, source, 1 - source_slot, destination, bytes, stat)
    end subroutine transfer_batch_matrix

    subroutine batch_copy(self, slot, source, destination, bytes, stat)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: slot
        type(c_ptr), intent(in) :: source, destination
        integer(c_size_t), intent(in) :: bytes
        type(status_t), intent(out) :: stat

        if (slot == 1) then
            call cuda_qwen35_copy_device(self%cuda_second, source, destination, bytes, stat)
        else
            call cuda_qwen35_copy_device(self%cuda, source, destination, bytes, stat)
        end if
    end subroutine batch_copy

    subroutine batch_add(self, slot, left, right, output, elements, stat)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: slot
        type(c_ptr), intent(in) :: left, right, output
        integer(c_size_t), intent(in) :: elements
        type(status_t), intent(out) :: stat

        if (slot == 1) then
            call cuda_qwen35_add_matrix_device(self%cuda_second, left, right, output, elements, stat)
        else
            call cuda_qwen35_add_matrix_device(self%cuda, left, right, output, elements, stat)
        end if
    end subroutine batch_add

    subroutine batch_rms_norm(self, slot, input, weights, output, hidden, batch, epsilon, stat)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: slot, batch
        type(c_ptr), intent(in) :: input, weights, output
        integer(c_size_t), intent(in) :: hidden
        real(real32), intent(in) :: epsilon
        type(status_t), intent(out) :: stat

        if (slot == 1) then
            call cuda_qwen35_rms_norm_matrix_device(self%cuda_second, input, weights, output, hidden, batch, epsilon, stat)
        else
            call cuda_qwen35_rms_norm_matrix_device(self%cuda, input, weights, output, hidden, batch, epsilon, stat)
        end if
    end subroutine batch_rms_norm

    subroutine batch_silu(self, slot, gate, up, elements, stat)
        class(qwen35_cpu_model_t), intent(in) :: self
        integer, intent(in) :: slot
        type(c_ptr), intent(in) :: gate, up
        integer(c_size_t), intent(in) :: elements
        type(status_t), intent(out) :: stat

        if (slot == 1) then
            call cuda_qwen35_silu_product_matrix_device(self%cuda_second, gate, up, elements, stat)
        else
            call cuda_qwen35_silu_product_matrix_device(self%cuda, gate, up, elements, stat)
        end if
    end subroutine batch_silu

    subroutine batch_project_one(self, slot, tensor_index, activation, activation_width, batch, output, output_width, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: slot, tensor_index, batch
        type(c_ptr), intent(in) :: activation, output
        integer(c_size_t), intent(in) :: activation_width, output_width
        type(status_t), intent(out) :: stat

        call stat%clear()
        if (slot < 0 .or. slot > 1) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA batch device slot')
            return
        end if
        if (batch_tensor_ready_on_slot(self, tensor_index, slot)) then
            if (self%file%tensors(tensor_index)%value_type == GGML_TYPE_Q8_0) then
                if (slot == 0) then
                    call cuda_q8_matmul_device_f32(self%cuda, self%cuda_weights(tensor_index), activation, &
                        activation_width * int(batch, c_size_t), batch, output, output_width * int(batch, c_size_t), stat)
                else
                    call cuda_q8_matmul_device_f32(self%cuda_second, self%cuda_weights_second(tensor_index), activation, &
                        activation_width * int(batch, c_size_t), batch, output, output_width * int(batch, c_size_t), stat)
                end if
            else if (slot == 0) then
                call cuda_q4_matmul_device_one_slot(self%cuda_q4, slot, self%cuda_q4_weights(tensor_index), activation, &
                    activation_width * int(batch, c_size_t), batch, output, output_width * int(batch, c_size_t), stat)
            else
                call cuda_q4_matmul_device_one_slot(self%cuda_q4, slot, self%cuda_q4_weights(tensor_index), activation, &
                    activation_width * int(batch, c_size_t), batch, output, output_width * int(batch, c_size_t), stat)
            end if
        else
            call stat%set(FORTAI_UNSUPPORTED, 'unsupported batched projection tensor')
        end if
    end subroutine batch_project_one

    subroutine batch_project_pair(self, slot, first_index, second_index, activation, activation_width, batch, &
            first_output, first_width, second_output, second_width, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: slot, first_index, second_index, batch
        type(c_ptr), intent(in) :: activation, first_output, second_output
        integer(c_size_t), intent(in) :: activation_width, first_width, second_width
        type(status_t), intent(out) :: stat
        integer(c_size_t) :: activation_total, first_total, second_total

        call stat%clear()
        if (slot < 0 .or. slot > 1) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA batch device slot')
            return
        end if
        activation_total = activation_width * int(batch, c_size_t)
        first_total = first_width * int(batch, c_size_t)
        second_total = second_width * int(batch, c_size_t)
        if (batch_tensor_ready_on_slot(self, first_index, slot) .and. &
            batch_tensor_ready_on_slot(self, second_index, slot) .and. &
            self%file%tensors(first_index)%value_type == GGML_TYPE_Q8_0 .and. &
            self%file%tensors(second_index)%value_type == GGML_TYPE_Q8_0) then
            if (slot == 0) then
                call cuda_q8_matmul_device_f32(self%cuda, self%cuda_weights(first_index), activation, activation_total, &
                    batch, first_output, first_total, stat)
            else
                call cuda_q8_matmul_device_f32(self%cuda_second, self%cuda_weights_second(first_index), activation, &
                    activation_total, batch, first_output, first_total, stat)
            end if
            if (.not. stat%is_ok()) return
            if (slot == 0) then
                call cuda_q8_matmul_device_f32(self%cuda, self%cuda_weights(second_index), activation, activation_total, &
                    batch, second_output, second_total, stat)
            else
                call cuda_q8_matmul_device_f32(self%cuda_second, self%cuda_weights_second(second_index), activation, &
                    activation_total, batch, second_output, second_total, stat)
            end if
        else if (batch_tensor_ready_on_slot(self, first_index, slot) .and. &
                batch_tensor_ready_on_slot(self, second_index, slot) .and. &
                is_q4_xl_type(self%file%tensors(first_index)%value_type) .and. &
                is_q4_xl_type(self%file%tensors(second_index)%value_type)) then
            call cuda_q4_matmul_device_pair_slot(self%cuda_q4, slot, self%cuda_q4_weights(first_index), &
                self%cuda_q4_weights(second_index), activation, activation_total, batch, first_output, first_total, &
                second_output, second_total, stat)
        else
            call batch_project_one(self, slot, first_index, activation, activation_width, batch, first_output, first_width, stat)
            if (.not. stat%is_ok()) return
            call batch_project_one(self, slot, second_index, activation, activation_width, batch, second_output, second_width, stat)
        end if
    end subroutine batch_project_pair

    subroutine batch_project_quad(self, slot, first_index, second_index, third_index, fourth_index, activation, &
            activation_width, batch, first_output, first_width, second_output, second_width, third_output, third_width, &
            fourth_output, fourth_width, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: slot, first_index, second_index, third_index, fourth_index, batch
        type(c_ptr), intent(in) :: activation, first_output, second_output, third_output, fourth_output
        integer(c_size_t), intent(in) :: activation_width, first_width, second_width, third_width, fourth_width
        type(status_t), intent(out) :: stat
        integer(c_size_t) :: activation_total

        call stat%clear()
        activation_total = activation_width * int(batch, c_size_t)
        if (slot >= 0 .and. slot <= 1 .and. &
            batch_tensor_ready_on_slot(self, first_index, slot) .and. &
            batch_tensor_ready_on_slot(self, second_index, slot) .and. &
            batch_tensor_ready_on_slot(self, third_index, slot) .and. &
            batch_tensor_ready_on_slot(self, fourth_index, slot) .and. &
            is_q4_xl_type(self%file%tensors(first_index)%value_type) .and. &
            is_q4_xl_type(self%file%tensors(second_index)%value_type) .and. &
            is_q4_xl_type(self%file%tensors(third_index)%value_type) .and. &
            is_q4_xl_type(self%file%tensors(fourth_index)%value_type)) then
            call cuda_q4_matmul_device_quad_slot(self%cuda_q4, slot, self%cuda_q4_weights(first_index), &
                self%cuda_q4_weights(second_index), self%cuda_q4_weights(third_index), &
                self%cuda_q4_weights(fourth_index), activation, activation_total, batch, first_output, &
                first_width * int(batch, c_size_t), second_output, second_width * int(batch, c_size_t), third_output, &
                third_width * int(batch, c_size_t), fourth_output, fourth_width * int(batch, c_size_t), stat)
            return
        end if
        call batch_project_pair(self, slot, first_index, second_index, activation, activation_width, batch, &
            first_output, first_width, second_output, second_width, stat)
        if (.not. stat%is_ok()) return
        call batch_project_pair(self, slot, third_index, fourth_index, activation, activation_width, batch, &
            third_output, third_width, fourth_output, fourth_width, stat)
    end subroutine batch_project_quad

    subroutine batch_ffn(self, slot, gate_index, up_index, down_index, activation, hidden_width, batch, &
            output, output_width, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: slot, gate_index, up_index, down_index, batch
        type(c_ptr), intent(in) :: activation, output
        integer(c_size_t), intent(in) :: hidden_width, output_width
        type(status_t), intent(out) :: stat
        integer(c_size_t) :: activation_total, output_total
        type(c_ptr) :: gate_output, up_output

        call stat%clear()
        if (slot < 0 .or. slot > 1 .or. batch <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA batch FFN arguments')
            return
        end if
        activation_total = hidden_width * int(batch, c_size_t)
        output_total = output_width * int(batch, c_size_t)
        ! Prompt batches have arbitrary shapes and already use a bounded-memory
        ! pair/SwiGLU/down route.  Retain fused plans only for scalar/MTP
        ! generation, where the fixed depth-two verifier needs at most 3 rows.
        if (batch <= 3 .and. batch_tensor_ready_on_slot(self, gate_index, slot) .and. &
            batch_tensor_ready_on_slot(self, up_index, slot) .and. &
            batch_tensor_ready_on_slot(self, down_index, slot) .and. &
            is_q4_xl_type(self%file%tensors(gate_index)%value_type) .and. &
            is_q4_xl_type(self%file%tensors(up_index)%value_type) .and. &
            is_q4_xl_type(self%file%tensors(down_index)%value_type)) then
            call cuda_q4_matmul_device_swiglu_down_slot(self%cuda_q4, slot, self%cuda_q4_weights(gate_index), &
                self%cuda_q4_weights(up_index), self%cuda_q4_weights(down_index), activation, activation_total, batch, &
                output, output_total, stat)
            return
        end if
        if (slot == 0) then
            gate_output = self%cuda_batch_ffn_gate
            up_output = self%cuda_batch_ffn_up
        else
            gate_output = self%cuda_batch_ffn_gate_second
            up_output = self%cuda_batch_ffn_up_second
        end if
        call batch_project_pair(self, slot, gate_index, up_index, activation, hidden_width, batch, &
            gate_output, int(self%feed_forward_size, c_size_t), up_output, &
            int(self%feed_forward_size, c_size_t), stat)
        if (.not. stat%is_ok()) return
        call batch_silu(self, slot, gate_output, up_output, &
            int(self%feed_forward_size, c_size_t) * int(batch, c_size_t), stat)
        if (.not. stat%is_ok()) return
        call batch_project_one(self, slot, down_index, gate_output, &
            int(self%feed_forward_size, c_size_t), batch, output, output_width, stat)
    end subroutine batch_ffn

    subroutine batch_project_triplet(self, slot, first_index, second_index, third_index, activation, activation_width, batch, &
            first_output, first_width, second_output, second_width, third_output, third_width, stat)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer, intent(in) :: slot, first_index, second_index, third_index, batch
        type(c_ptr), intent(in) :: activation, first_output, second_output, third_output
        integer(c_size_t), intent(in) :: activation_width, first_width, second_width, third_width
        type(status_t), intent(out) :: stat
        integer(c_size_t) :: activation_total, first_total, second_total, third_total

        call stat%clear()
        if (slot < 0 .or. slot > 1) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA batch device slot')
            return
        end if
        activation_total = activation_width * int(batch, c_size_t)
        first_total = first_width * int(batch, c_size_t)
        second_total = second_width * int(batch, c_size_t)
        third_total = third_width * int(batch, c_size_t)
        if (batch_tensor_ready_on_slot(self, first_index, slot) .and. &
            batch_tensor_ready_on_slot(self, second_index, slot) .and. &
            batch_tensor_ready_on_slot(self, third_index, slot) .and. &
            self%file%tensors(first_index)%value_type == GGML_TYPE_Q8_0 .and. &
            self%file%tensors(second_index)%value_type == GGML_TYPE_Q8_0 .and. &
            self%file%tensors(third_index)%value_type == GGML_TYPE_Q8_0) then
            if (slot == 0) then
                call cuda_q8_matmul_device_f32(self%cuda, self%cuda_weights(first_index), activation, activation_total, &
                    batch, first_output, first_total, stat)
            else
                call cuda_q8_matmul_device_f32(self%cuda_second, self%cuda_weights_second(first_index), activation, &
                    activation_total, batch, first_output, first_total, stat)
            end if
            if (.not. stat%is_ok()) return
            if (slot == 0) then
                call cuda_q8_matmul_device_f32(self%cuda, self%cuda_weights(second_index), activation, activation_total, &
                    batch, second_output, second_total, stat)
            else
                call cuda_q8_matmul_device_f32(self%cuda_second, self%cuda_weights_second(second_index), activation, &
                    activation_total, batch, second_output, second_total, stat)
            end if
            if (.not. stat%is_ok()) return
            if (slot == 0) then
                call cuda_q8_matmul_device_f32(self%cuda, self%cuda_weights(third_index), activation, activation_total, &
                    batch, third_output, third_total, stat)
            else
                call cuda_q8_matmul_device_f32(self%cuda_second, self%cuda_weights_second(third_index), activation, &
                    activation_total, batch, third_output, third_total, stat)
            end if
        else if (batch_tensor_ready_on_slot(self, first_index, slot) .and. &
                batch_tensor_ready_on_slot(self, second_index, slot) .and. &
                batch_tensor_ready_on_slot(self, third_index, slot) .and. &
                is_q4_xl_type(self%file%tensors(first_index)%value_type) .and. &
                is_q4_xl_type(self%file%tensors(second_index)%value_type) .and. &
                is_q4_xl_type(self%file%tensors(third_index)%value_type)) then
            call cuda_q4_matmul_device_triplet_slot(self%cuda_q4, slot, self%cuda_q4_weights(first_index), &
                self%cuda_q4_weights(second_index), self%cuda_q4_weights(third_index), activation, activation_total, &
                batch, first_output, first_total, second_output, second_total, third_output, third_total, stat)
        else
            call batch_project_one(self, slot, first_index, activation, activation_width, batch, first_output, first_width, stat)
            if (.not. stat%is_ok()) return
            call batch_project_one(self, slot, second_index, activation, activation_width, batch, second_output, second_width, stat)
            if (.not. stat%is_ok()) return
            call batch_project_one(self, slot, third_index, activation, activation_width, batch, third_output, third_width, stat)
        end if
    end subroutine batch_project_triplet

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
        if (length <= 0) call get_environment_variable('LLAMA_ARG_MODEL_DRAFT', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_DRAFT_MODEL', value, length=length)
        if (length > 0) then
            ! The published Qwen3.8 sidecar is a tensor carrier, not a
            ! standalone transformer.  Its filename is the only portable
            ! signal available to the native CLI when --model-draft is used.
            length = min(length, len(value))
            native_mtp_requested = index(value(1:length), 'mtp') > 0 .or. &
                index(value(1:length), 'MTP') > 0
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

    logical function native_mtp_trace_enabled()
        character(len=8) :: value
        integer :: length
        integer, save :: cached = -1

        if (cached < 0) then
            value = ''
            call get_environment_variable('FORTAI_TRACE_MTP', value, length=length)
            cached = 0
            if (length > 0 .and. length <= len(value)) then
                if (value(:length) /= '0' .and. value(:length) /= 'false' .and. &
                    value(:length) /= 'off') cached = 1
            end if
        end if
        native_mtp_trace_enabled = cached == 1
    end function native_mtp_trace_enabled

    subroutine qwen35_cpu_forward_greedy(self, token_id, position, next_token, logit_sum, stat, use_mtp, &
            save_mtp_hidden, compute_logits)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id, position
        integer(int64), intent(out) :: next_token
        real(real32), intent(out) :: logit_sum
        type(status_t), intent(out) :: stat
        logical, intent(in), optional :: use_mtp
        logical, intent(in), optional :: save_mtp_hidden
        logical, intent(in), optional :: compute_logits
        integer(c_int) :: code, next_token_c
        integer(int64) :: device_token
        integer(int64) :: verify_input(1), verified_token(1)
        logical :: should_use_mtp, should_save_mtp, should_compute_logits

        next_token = 0_int64
        logit_sum = 0.0_real32
        should_use_mtp = .true.
        if (present(use_mtp)) should_use_mtp = use_mtp
        should_save_mtp = should_use_mtp
        if (present(save_mtp_hidden)) should_save_mtp = save_mtp_hidden
        should_compute_logits = .true.
        if (present(compute_logits)) should_compute_logits = compute_logits
        device_token = -1_int64
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
            if (self%batch_supported(1) .and. allocated(self%mtp_verify_hidden)) then
                if (self%mtp_active .and. should_use_mtp .and. position > 0_int64) then
                    call qwen35_cpu_mtp_pair(self, token_id, position, self%mtp_pending_hidden, stat)
                    if (.not. stat%is_ok()) return
                end if
                verify_input(1) = token_id
                call self%forward_batch_verify(verify_input, position, verified_token, stat)
                if (.not. stat%is_ok()) return
                next_token = verified_token(1)
                if (self%mtp_active .and. should_save_mtp) then
                    self%mtp_pending_hidden = self%mtp_verify_hidden(:self%hidden_size)
                    self%mtp_last_target_position = position
                end if
                return
            end if
            call qwen35_cpu_forward_body(self, token_id, position, self%logits, stat, .false., should_use_mtp, &
                should_save_mtp, should_compute_logits, greedy_only=.true., greedy_token=device_token)
            if (.not. stat%is_ok()) return
            if (.not. should_compute_logits) return
            if (device_token >= 0_int64) then
                next_token = device_token
                return
            end if
            block
                integer :: index
                call self%cuda%argmax_device(self%cuda_logits, int(self%vocabulary_size, c_size_t), index, stat)
                if (.not. stat%is_ok()) return
                next_token = int(index, int64)
            end block
            return
        end if
        call qwen35_cpu_forward(self, token_id, position, self%logits, stat, .true., should_use_mtp, should_save_mtp, &
            should_compute_logits)
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
        integer :: i, capacity, layer_slot, accepted_trace
        integer(int64) :: draft_token, draft_token_second
        integer(int64) :: verify_input(3), verified_tokens(3)
        real(real32) :: draft_sum

        call stat%clear()
        count = 0
        logit_sum = 0.0_real32
        if (size(tokens) == 0) then
            call stat%set(FORTAI_INVALID, 'speculative output workspace is empty')
            return
        end if
        capacity = min(size(tokens), size(c_tokens))
        if (.not. self%fast_enabled .and. self%mtp_active) then
            if (native_mtp_limit() >= 2 .and. capacity >= 3 .and. &
                position + 2_int64 < self%max_context .and. self%batch_supported(3)) then
                ! A draft depth of two means CURRENT + DRAFT1 + DRAFT2 is
                ! verified in one target batch.  llama.cpp uses this same
                ! three-row contract; treating n_max=2 as one draft leaves
                ! substantial generation throughput unused.
                call self%mtp_draft_greedy(token_id, position, draft_token, draft_sum, stat)
                if (.not. stat%is_ok()) return
                call qwen35_cpu_mtp_continue_cuda(self, draft_token, position + 1_int64, &
                    draft_token_second, stat)
                if (.not. stat%is_ok()) return
                verify_input = [token_id, draft_token, draft_token_second]
                call self%forward_batch_verify(verify_input, position, verified_tokens, stat)
                if (.not. stat%is_ok()) return
                call qwen35_cpu_mtp_refresh_verify_cuda(self, verify_input, position, stat)
                if (.not. stat%is_ok()) return
                self%mtp_last_draft_token = draft_token_second
                self%mtp_last_draft_match = draft_token == verified_tokens(1) .and. &
                    draft_token_second == verified_tokens(2)
                if (draft_token /= verified_tokens(1)) then
                    accepted_trace = 0
                else if (draft_token_second /= verified_tokens(2)) then
                    accepted_trace = 1
                else
                    accepted_trace = 2
                end if
                if (native_mtp_trace_enabled()) then
                    write(error_unit, '(a,i0,6(a,i0))') 'fortai-mtp: position=', position, &
                        ' draft1=', draft_token, ' target1=', verified_tokens(1), &
                        ' draft2=', draft_token_second, ' target2=', verified_tokens(2), &
                        ' bonus=', verified_tokens(3), ' accepted=', accepted_trace
                end if
                if (draft_token /= verified_tokens(1)) then
                    call self%accept_sampled_speculative(verify_input(2:3), position, 0, stat)
                    if (.not. stat%is_ok()) return
                    tokens(1) = verified_tokens(1)
                    count = 1
                else if (draft_token_second /= verified_tokens(2)) then
                    call self%accept_sampled_speculative(verify_input(2:3), position, 1, stat)
                    if (.not. stat%is_ok()) return
                    tokens(1:2) = [draft_token, verified_tokens(2)]
                    count = 2
                else
                    call self%accept_sampled_speculative(verify_input(2:3), position, 2, stat)
                    if (.not. stat%is_ok()) return
                    tokens(1:3) = [draft_token, draft_token_second, verified_tokens(3)]
                    count = 3
                end if
                logit_sum = 0.0_real32
                return
            end if
            if (native_mtp_limit() >= 2 .and. capacity >= 2 .and. &
                position + 1_int64 < self%max_context .and. self%batch_supported(2)) then
                ! Pair the unevaluated current token with the preceding target
                ! hidden row.  The MTP head proposes the next token before the
                ! target advances, matching llama.cpp's draft/process order.
                call self%mtp_draft_greedy(token_id, position, draft_token, draft_sum, stat)
                if (.not. stat%is_ok()) return
                verify_input(:2) = [token_id, draft_token]
                call self%forward_batch_verify(verify_input(:2), position, verified_tokens(:2), stat)
                if (.not. stat%is_ok()) return
                self%mtp_last_draft_token = draft_token
                self%mtp_last_draft_match = draft_token == verified_tokens(1)
                if (native_mtp_trace_enabled()) then
                    write(error_unit, '(a,i0,3(a,i0))') 'fortai-mtp: position=', position, &
                        ' draft1=', draft_token, ' target1=', verified_tokens(1), ' accepted=', &
                        merge(1, 0, self%mtp_last_draft_match)
                end if
                if (self%mtp_last_draft_match) then
                    tokens(1) = draft_token
                    tokens(2) = verified_tokens(2)
                    count = 2
                    self%mtp_pending_hidden = self%mtp_verify_hidden(self%hidden_size + 1:2 * self%hidden_size)
                    self%mtp_last_target_position = position + 1_int64
                    ! The draft cache consumed CURRENT to make DRAFT_TOKEN.
                    ! Commit the accepted token with the target hidden row for
                    ! CURRENT so the next MTP step starts from the same prefix.
                    call qwen35_cpu_mtp_pair(self, draft_token, position + 1_int64, &
                        self%mtp_verify_hidden(:self%hidden_size), stat)
                    if (.not. stat%is_ok()) return
                else
                    do i = 1, self%layer_count
                        if (.not. self%layers(i)%recurrent) cycle
                        layer_slot = batch_layer_device(self, i)
                        if (layer_slot == 0) then
                            call self%layers(i)%cuda_recurrent%restore_first(stat)
                        else
                            call self%layers(i)%cuda_recurrent_second%restore_first(stat)
                        end if
                        if (.not. stat%is_ok()) return
                    end do
                    ! The batched recurrent kernels checkpoint their state
                    ! immediately after token zero.  Restore that accepted
                    ! prefix and discard only the rejected speculative row;
                    ! attention KV row position+1 is overwritten by the next
                    ! committed target call.
                    tokens(1) = verified_tokens(1)
                    count = 1
                    self%mtp_pending_hidden = self%mtp_verify_hidden(:self%hidden_size)
                    self%mtp_last_target_position = position
                end if
                logit_sum = 0.0_real32
                return
            end if
        end if
        if (.not. self%fast_enabled .and. self%mtp_active .and. native_mtp_limit() >= 2 .and. &
            capacity >= 2 .and. position + 1_int64 < self%max_context) then
            call self%forward_greedy(token_id, position, tokens(1), logit_sum, stat)
            if (.not. stat%is_ok()) return
            call self%mtp_draft_greedy(tokens(1), position + 1_int64, draft_token, draft_sum, stat)
            if (.not. stat%is_ok()) return
            call self%forward_greedy(tokens(1), position + 1_int64, tokens(2), draft_sum, stat)
            if (.not. stat%is_ok()) return
            self%mtp_last_draft_token = draft_token
            self%mtp_last_draft_match = draft_token == tokens(2)
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

    subroutine qwen35_cpu_forward_body(self, token_id, position, logits, stat, download_logits, use_mtp, &
            save_mtp_hidden, compute_logits, greedy_only, greedy_token)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id, position
        real(real32), contiguous, intent(out) :: logits(:)
        type(status_t), intent(out) :: stat
        logical, intent(in), optional :: download_logits
        logical, intent(in), optional :: use_mtp
        logical, intent(in), optional :: save_mtp_hidden
        logical, intent(in), optional :: compute_logits
        logical, intent(in), optional :: greedy_only
        integer(int64), intent(out), optional :: greedy_token
        integer :: i, layer_start, segment_graph_index, segment_graph_slot
        logical :: capture_graph, capture_segment
        logical :: should_download, should_use_mtp, should_save_mtp, should_compute_logits, should_greedy_device

        call stat % clear()
        should_download = .true.
        if (present(download_logits)) should_download = download_logits
        should_use_mtp = .true.
        if (present(use_mtp)) should_use_mtp = use_mtp
        should_save_mtp = should_use_mtp
        if (present(save_mtp_hidden)) should_save_mtp = save_mtp_hidden
        should_compute_logits = .true.
        if (present(compute_logits)) should_compute_logits = compute_logits
        should_greedy_device = .false.
        if (present(greedy_only)) should_greedy_device = greedy_only
        if (present(greedy_token)) greedy_token = -1_int64
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
        if (self%mtp_active .and. should_use_mtp) then
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
        capture_segment = .false.
        layer_start = 1
        segment_graph_index = merge(2, 1, position >= 4096_int64)
        segment_graph_slot = merge(62, 0, position >= 4096_int64)
        ! llama.cpp's scheduler keeps one reusable graph for each contiguous
        ! device segment and leaves cross-device bridges outside capture.  A
        ! split Q4 model has exactly that shape: capture the primary prefix
        ! on the GGML scheduler stream, then execute the remote suffix with
        ! its ordinary event/bridge schedule.  The first decode warms every
        ! plan; the second captures stable pointers; later tokens replay it.
        if (self%cuda_device_pipeline .and. self%cuda_segment_graph_enabled .and. &
            should_compute_logits .and. position > 0_int64) then
            if (self%cuda_segment_graph_ready(segment_graph_index)) then
                call self%cuda%graph_launch_slot(segment_graph_slot, stat)
                if (.not. stat%is_ok()) return
                layer_start = self%cuda_segment_graph_end + 1
            else if (self%cuda_segment_graph_warmup(segment_graph_index)) then
                call self%cuda%capture_begin_slot(segment_graph_slot, stat)
                if (stat%is_ok()) then
                    capture_segment = .true.
                else
                    ! Capture is an optional launch optimization.  If a
                    ! driver or a newly introduced kernel rejects it, keep
                    ! the proven stream/event path for the rest of the run.
                    call stat%clear()
                    self%cuda_segment_graph_enabled = .false.
                    self%cuda_segment_graph_warmup(segment_graph_index) = .false.
                end if
            else
                self%cuda_segment_graph_warmup(segment_graph_index) = .true.
            end if
        end if
        ! Prompt evaluation can skip logits work for intermediate tokens.  The
        ! final token still runs the output projection and records the hidden
        ! state needed by the native MTP verifier.
        if (self%cuda_device_pipeline .and. self%cuda_graph_enabled .and. should_compute_logits) then
            if (self%cuda_graph_ready) then
                call self%cuda%graph_launch(stat)
                if (.not. stat%is_ok()) return
            else if (position > 0_int64) then
                call self%cuda%capture_begin(stat)
                if (.not. stat%is_ok()) return
                capture_graph = .true.
            end if
        end if

        if (capture_segment) then
            do i = 1, self%cuda_segment_graph_end
                if (.not. cuda_layer_device_ready(self, i)) then
                    call stat%set(FORTAI_UNSUPPORTED, 'CUDA segment graph layer became unavailable')
                    return
                end if
                if (self%layers(i)%recurrent) then
                    call forward_recurrent_device(self, i, stat)
                else
                    call forward_attention_device(self, i, position, stat)
                end if
                if (.not. stat%is_ok()) return
            end do
            call self%cuda%capture_end_slot(segment_graph_slot, stat)
            if (.not. stat%is_ok()) then
                self%cuda_segment_graph_enabled = .false.
                self%cuda_segment_graph_warmup(segment_graph_index) = .false.
                call stat%clear()
                capture_segment = .false.
                layer_start = 1
            else
                self%cuda_segment_graph_ready(segment_graph_index) = .true.
                call self%cuda%graph_launch_slot(segment_graph_slot, stat)
                if (.not. stat%is_ok()) return
                layer_start = self%cuda_segment_graph_end + 1
            end if
        end if

        if (.not. self%cuda_device_pipeline .or. .not. self%cuda_graph_ready .or. &
            .not. self%cuda_graph_enabled .or. .not. should_compute_logits) then
            do i = layer_start, self % layer_count
                if (self%cuda_device_pipeline) then
                    if (cuda_layer_device_ready(self, i)) then
                        if (self%layers(i)%recurrent) then
                            call forward_recurrent_device(self, i, stat)
                            if (.not. stat%is_ok()) return
                            cycle
                        end if
                        call forward_attention_device(self, i, position, stat)
                        if (.not. stat%is_ok()) return
                        cycle
                    end if
                    ! Keep the resident pipeline useful when one large
                    ! attention cache cannot fit on the selected CUDA device:
                    ! synchronize only this boundary and run that layer through
                    ! the independent host cache/oracle before uploading the
                    ! hidden state back for the next resident layer.
                    call self%cuda%download_real(self%cuda_x, self%x, stat)
                    if (.not. stat%is_ok()) return
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
                if (self%cuda_device_pipeline) then
                    call self%cuda%upload_real(self%cuda_x, self%x, stat)
                    if (.not. stat%is_ok()) return
                end if
            end do
        end if

        if (self%cuda_device_pipeline .and. should_compute_logits .and. (.not. self%cuda_graph_ready .or. &
            .not. self%cuda_graph_enabled)) then
            if (should_greedy_device .and. present(greedy_token)) then
                call forward_output_device_greedy(self, greedy_token, stat)
            else
                call forward_output_device(self, stat)
            end if
            if (.not. stat%is_ok()) return
        end if

        if (capture_graph) then
            call self%cuda%capture_end(stat)
            if (.not. stat%is_ok()) return
            self%cuda_graph_ready = .true.
            call self%cuda%graph_launch(stat)
            if (.not. stat%is_ok()) return
        end if

        ! Native MTP's NextN head is intentionally host-controlled, but the
        ! target transformer remains device-resident.  Preserve the final
        ! target hidden state before returning the CUDA logits so the next
        ! token's host-side MTP verification can consume it without forcing
        ! the whole target decode back through the host-boundary path.  This
        ! download must happen after graph capture has ended.
        if (self%cuda_device_pipeline .and. self%mtp_active .and. should_save_mtp) then
            call self%cuda%download_real(self%cuda_x, self%x, stat)
            if (.not. stat%is_ok()) return
            self%mtp_pending_hidden = self%x
            self%mtp_last_target_position = position
        end if

        if (self%cuda_device_pipeline) then
            ! A graph that already produced logits still needs only the
            ! backend-side argmax for greedy decode.  The remote-output helper
            ! sets greedy_token itself; the negative sentinel selects the
            ! primary-device fallback for single-device graphs or Q8 output.
            if (should_compute_logits .and. should_greedy_device .and. present(greedy_token)) then
                if (greedy_token < 0_int64) then
                    block
                        integer :: index
                        call self%cuda%argmax_device(self%cuda_logits, int(self%vocabulary_size, c_size_t), &
                            index, stat)
                        if (.not. stat%is_ok()) return
                        greedy_token = int(index, int64)
                    end block
                end if
            end if
            if (should_compute_logits .and. should_download) then
                call self%cuda%download_real(self%cuda_logits, logits, stat)
                if (.not. stat%is_ok()) return
            end if
            return
        end if
        if (.not. should_compute_logits) return
        call rms_norm(self%x, self%file%tensors(self%output_norm), self%norm_epsilon, &
            self % normalized, stat)
        if (.not. stat % is_ok()) return
        call model_matvec(self, self % output, self % normalized, logits, stat)
        if (stat%is_ok() .and. self%mtp_active .and. should_save_mtp) then
            self%mtp_pending_hidden = self%normalized
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
        if (size(hidden) /= self%hidden_size .or. token_id < 0_int64 .or. &
            token_id >= self%vocabulary_size .or. position < 0_int64 .or. &
            position >= self%max_context) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 native MTP pair dimensions are invalid')
            return
        end if
        if (self%mtp_cuda_slot >= 0) then
            self%mtp_target_hidden = hidden
            call qwen35_cpu_mtp_prepare_inputs_device(self, token_id, stat, hidden)
            if (.not. stat%is_ok()) return
            call qwen35_cpu_mtp_pair_cuda(self, position, stat, .true.)
            if (.not. stat%is_ok()) return
            self%x = self%mtp_target_hidden
            self%mtp_last_pair_position = position
            self%mtp_last_pair_token = token_id
            return
        end if
        call allocate_attention_cache(self, self%mtp_layer)
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

    subroutine qwen35_cpu_mtp_prepare_inputs_device(self, token_id, stat, hidden, preserve_hidden, hidden_device)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: token_id
        type(status_t), intent(out) :: stat
        real(real32), contiguous, intent(in), optional :: hidden(:)
        logical, intent(in), optional :: preserve_hidden
        type(c_ptr), intent(in), optional :: hidden_device
        integer(c_int32_t), target :: token_c(1)
        integer(c_size_t) :: hidden_elements
        integer :: embed_index, slot, hidden_sources
        logical :: keep_hidden
        type(c_ptr) :: batch_x, batch_residual

        call stat%clear()
        slot = self%mtp_cuda_slot
        keep_hidden = .false.
        if (present(preserve_hidden)) keep_hidden = preserve_hidden
        hidden_sources = 0
        if (present(hidden)) hidden_sources = hidden_sources + 1
        if (keep_hidden) hidden_sources = hidden_sources + 1
        if (present(hidden_device)) then
            if (c_associated(hidden_device)) hidden_sources = hidden_sources + 1
        end if
        if (slot < 0 .or. token_id < 0_int64 .or. token_id >= self%vocabulary_size .or. &
            hidden_sources /= 1) then
            call stat%set(FORTAI_INVALID, 'invalid Qwen3.5 MTP device inputs')
            return
        end if
        if (present(hidden)) then
            if (size(hidden) /= self%hidden_size) then
                call stat%set(FORTAI_INVALID, 'invalid Qwen3.5 MTP device hidden row')
                return
            end if
        end if
        hidden_elements = int(self%hidden_size, c_size_t)
        if (slot == 0) then
            batch_x = self%cuda_batch_x
            batch_residual = self%cuda_batch_residual
        else
            batch_x = self%cuda_batch_x_second
            batch_residual = self%cuda_batch_residual_second
        end if
        if (keep_hidden) then
            call batch_copy(self, slot, batch_x, batch_residual, &
                hidden_elements * int(storage_size(self%x(1)) / 8, c_size_t), stat)
        else if (present(hidden_device)) then
            call batch_copy(self, slot, hidden_device, batch_residual, &
                hidden_elements * int(storage_size(self%x(1)) / 8, c_size_t), stat)
        else if (slot == 0) then
            call self%cuda%upload_real(batch_residual, hidden, stat)
        else
            call self%cuda_second%upload_real(batch_residual, hidden, stat)
        end if
        if (.not. stat%is_ok()) return

        token_c(1) = int(token_id, c_int32_t)
        embed_index = self%token_embedding
        if (self%mtp_embed_tokens > 0) embed_index = self%mtp_embed_tokens
        if (embed_index == self%token_embedding .and. &
            c_associated(self%cuda_mtp_embedding_weights%handle)) then
            call cuda_q4_embedding_device_batch_slot(self%cuda_q4, slot, &
                self%cuda_mtp_embedding_weights, token_c, 1, batch_x, hidden_elements, stat)
        else if (batch_tensor_ready_on_slot(self, embed_index, slot) .and. &
                self%file%tensors(embed_index)%value_type == GGML_TYPE_Q8_0) then
            if (slot == 0) then
                call cuda_qwen35_embedding_device_batch(self%cuda, self%cuda_weights(embed_index), &
                    token_c, 1, batch_x, hidden_elements, stat)
            else
                call cuda_qwen35_embedding_device_batch(self%cuda_second, &
                    self%cuda_weights_second(embed_index), token_c, 1, batch_x, hidden_elements, stat)
            end if
        else if (batch_tensor_ready_on_slot(self, embed_index, slot) .and. &
                is_q4_xl_type(self%file%tensors(embed_index)%value_type)) then
            call cuda_q4_embedding_device_batch_slot(self%cuda_q4, slot, &
                self%cuda_q4_weights(embed_index), token_c, 1, batch_x, hidden_elements, stat)
        else
            call self%file%tensors(embed_index)%get_row(token_id + 1_int64, self%mtp_embedding, stat)
            if (.not. stat%is_ok()) return
            if (slot == 0) then
                call self%cuda%upload_real(batch_x, self%mtp_embedding, stat)
            else
                call self%cuda_second%upload_real(batch_x, self%mtp_embedding, stat)
            end if
        end if
    end subroutine qwen35_cpu_mtp_prepare_inputs_device

    subroutine qwen35_cpu_mtp_pair_cuda(self, position, stat, inputs_ready)
        class(qwen35_cpu_model_t), intent(inout) :: self
        integer(int64), intent(in) :: position
        type(status_t), intent(out) :: stat
        logical, intent(in), optional :: inputs_ready
        integer(c_int) :: draft_index(1)
        integer(c_size_t) :: hidden, ffn, query, key, value, core
        integer(c_size_t) :: hidden_bytes
        integer :: slot
        logical :: ready, capture_mtp
        type(c_ptr) :: batch_x, batch_residual, batch_normalized, batch_hidden
        type(c_ptr) :: batch_qkv, batch_gate, batch_alpha, batch_beta
        type(c_ptr) :: batch_ffn_gate, batch_ffn_up, batch_q, batch_k, batch_v
        type(c_ptr) :: batch_attention, norm_attn, norm_post, spec_logits

        call stat%clear()
        ready = .false.
        if (present(inputs_ready)) ready = inputs_ready
        slot = self%mtp_cuda_slot
        hidden = int(self%hidden_size, c_size_t)
        ffn = int(self%feed_forward_size, c_size_t)
        query = int(2 * self%attention_heads * self%attention_head_size, c_size_t)
        key = int(self%attention_heads_kv * self%attention_head_size, c_size_t)
        value = int(self%attention_heads_kv * self%value_length, c_size_t)
        core = int(self%attention_heads * self%value_length, c_size_t)
        hidden_bytes = hidden * int(storage_size(self%x(1)) / 8, c_size_t)
        call select_batch_workspace(self, slot, self%layer_count, batch_x, batch_residual, &
            batch_normalized, batch_hidden, batch_qkv, batch_gate, batch_alpha, batch_beta, &
            batch_ffn_gate, batch_ffn_up, batch_q, batch_k, batch_v, batch_attention, &
            norm_attn, norm_post)
        if (.not. ready) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 MTP CUDA inputs are not prepared')
            return
        end if
        if (slot == 0) then
            spec_logits = self%cuda_spec_logits
        else
            spec_logits = self%cuda_spec_logits_second
        end if
        ! The MTP layer uses fixed resident buffers and weights.  Keep its
        ! position in device memory and capture the complete projection/layer/
        ! head sequence once, leaving only token preparation and the compact
        ! result download outside the graph.  Slot 63 is private to this graph;
        ! slots 64/65 hold the short/long target-verification regimes.
        mtp_compute: do
        if (slot == 0) then
            call self%cuda%set_position(int(position), stat)
        else
            call self%cuda_second%set_position(int(position), stat)
        end if
        if (.not. stat%is_ok()) return
        capture_mtp = .false.
        if (self%cuda_segment_graph_enabled .and. .not. self%cuda_mtp_graph_disabled) then
            if (self%cuda_mtp_graph_ready) then
                if (slot == 0) then
                    call self%cuda%graph_launch_slot(63, stat)
                else
                    call self%cuda_second%graph_launch_slot(63, stat)
                end if
                if (.not. stat%is_ok()) return
                exit mtp_compute
            else if (self%cuda_mtp_graph_warmup) then
                if (slot == 0) then
                    call self%cuda%capture_begin_slot(63, stat)
                else
                    call self%cuda_second%capture_begin_slot(63, stat)
                end if
                if (stat%is_ok()) then
                    capture_mtp = .true.
                else
                    call stat%clear()
                    self%cuda_mtp_graph_disabled = .true.
                end if
            else
                self%cuda_mtp_graph_warmup = .true.
            end if
        end if
        call batch_rms_norm(self, slot, batch_x, self%cuda_mtp_enorm, batch_normalized, &
            hidden, 1, self%norm_epsilon, stat)
        if (.not. stat%is_ok()) return
        call batch_rms_norm(self, slot, batch_residual, self%cuda_mtp_hnorm, batch_hidden, &
            hidden, 1, self%norm_epsilon, stat)
        if (.not. stat%is_ok()) return
        if (slot == 0) then
            call cuda_qwen35_concat_device(self%cuda, batch_normalized, batch_hidden, hidden, &
                batch_gate, stat)
        else
            call cuda_qwen35_concat_device(self%cuda_second, batch_normalized, batch_hidden, &
                hidden, batch_gate, stat)
        end if
        if (.not. stat%is_ok()) return
        call batch_project_one(self, slot, self%mtp_eh_proj, batch_gate, 2_c_size_t * hidden, &
            1, batch_x, hidden, stat)
        if (.not. stat%is_ok()) return

        call batch_copy(self, slot, batch_x, batch_residual, hidden_bytes, stat)
        if (.not. stat%is_ok()) return
        call batch_rms_norm(self, slot, batch_x, self%cuda_mtp_attn_norm, batch_normalized, &
            hidden, 1, self%norm_epsilon, stat)
        if (.not. stat%is_ok()) return
        call batch_project_triplet(self, slot, self%mtp_layer%attn_q, self%mtp_layer%attn_k, &
            self%mtp_layer%attn_v, batch_normalized, hidden, 1, batch_q, query, batch_k, &
            key, batch_v, value, stat)
        if (.not. stat%is_ok()) return
        call self%mtp_layer%cuda_attention%run_core_device_batch(batch_q, query, batch_k, key, &
            batch_v, value, int(position), 1, batch_attention, core, stat)
        if (.not. stat%is_ok()) return
        call batch_project_one(self, slot, self%mtp_layer%attn_out, batch_attention, core, &
            1, batch_hidden, hidden, stat)
        if (.not. stat%is_ok()) return
        call batch_add(self, slot, batch_hidden, batch_residual, batch_x, hidden, stat)
        if (.not. stat%is_ok()) return

        call batch_copy(self, slot, batch_x, batch_residual, hidden_bytes, stat)
        if (.not. stat%is_ok()) return
        call batch_rms_norm(self, slot, batch_x, self%cuda_mtp_post_norm, batch_normalized, &
            hidden, 1, self%norm_epsilon, stat)
        if (.not. stat%is_ok()) return
        call batch_ffn(self, slot, self%mtp_layer%ffn_gate, self%mtp_layer%ffn_up, self%mtp_layer%ffn_down, &
            batch_normalized, hidden, 1, batch_hidden, hidden, stat)
        if (.not. stat%is_ok()) return
        call batch_add(self, slot, batch_hidden, batch_residual, batch_x, hidden, stat)
        if (.not. stat%is_ok()) return
        call batch_rms_norm(self, slot, batch_x, self%cuda_mtp_head_norm, batch_normalized, &
            hidden, 1, self%norm_epsilon, stat)
        if (.not. stat%is_ok()) return
        call batch_project_one(self, slot, self%mtp_output, batch_normalized, hidden, 1, &
            spec_logits, int(self%vocabulary_size, c_size_t), stat)
        if (.not. stat%is_ok()) return
        if (capture_mtp) then
            if (slot == 0) then
                call self%cuda%capture_end_slot(63, stat)
            else
                call self%cuda_second%capture_end_slot(63, stat)
            end if
            if (.not. stat%is_ok()) then
                ! Capture is an optional scheduler optimization.  Retry this
                ! pair on the ordinary stream if a future kernel is not
                ! capturable, preserving the established correctness path.
                call stat%clear()
                self%cuda_mtp_graph_disabled = .true.
                cycle mtp_compute
            end if
            self%cuda_mtp_graph_ready = .true.
            if (slot == 0) then
                call self%cuda%graph_launch_slot(63, stat)
            else
                call self%cuda_second%graph_launch_slot(63, stat)
            end if
            if (.not. stat%is_ok()) return
        end if
        exit mtp_compute
    end do mtp_compute
    if (slot == 0) then
        call self%cuda%argmax_rows_device(spec_logits, int(self%vocabulary_size, c_size_t), &
            draft_index, stat)
    else
        call self%cuda_second%argmax_rows_device(spec_logits, &
            int(self%vocabulary_size, c_size_t), draft_index, stat)
    end if
    if (.not. stat%is_ok()) return
    self%mtp_device_draft_token = int(draft_index(1), int64)
end subroutine qwen35_cpu_mtp_pair_cuda

subroutine qwen35_cpu_mtp_continue_cuda(self, token_id, position, next_token, stat)
    class(qwen35_cpu_model_t), intent(inout) :: self
    integer(int64), intent(in) :: token_id, position
    integer(int64), intent(out) :: next_token
    type(status_t), intent(out) :: stat
    type(c_ptr) :: batch_x, batch_residual, batch_normalized, batch_hidden
    type(c_ptr) :: batch_qkv, batch_gate, batch_alpha, batch_beta
    type(c_ptr) :: batch_ffn_gate, batch_ffn_up, batch_q, batch_k, batch_v
    type(c_ptr) :: batch_attention, norm_attn, norm_post

    call stat%clear()
    next_token = -1_int64
    if (self%mtp_cuda_slot < 0 .or. token_id < 0_int64 .or. &
        token_id >= self%vocabulary_size .or. position < 0_int64 .or. &
        position >= self%max_context) then
        call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA MTP continuation is invalid')
        return
    end if
    call select_batch_workspace(self, self%mtp_cuda_slot, self%layer_count, batch_x, batch_residual, &
        batch_normalized, batch_hidden, batch_qkv, batch_gate, batch_alpha, batch_beta, &
        batch_ffn_gate, batch_ffn_up, batch_q, batch_k, batch_v, batch_attention, &
        norm_attn, norm_post)
    ! The recursive MTP input is its exported h_nextn row: the
    ! shared-head-normalized post-FFN state left in BATCH_NORMALIZED by
    ! qwen35_cpu_mtp_pair_cuda.  The raw BATCH_X residual is not the
    ! model's trained continuation feature.
    call qwen35_cpu_mtp_prepare_inputs_device(self, token_id, stat, hidden_device=batch_normalized)
    if (.not. stat%is_ok()) return
    call qwen35_cpu_mtp_pair_cuda(self, position, stat, .true.)
    if (.not. stat%is_ok()) return
    self%mtp_last_pair_position = position
    self%mtp_last_pair_token = token_id
    next_token = self%mtp_device_draft_token
end subroutine qwen35_cpu_mtp_continue_cuda

subroutine qwen35_cpu_mtp_refresh_verify_cuda(self, token_ids, position, stat)
    class(qwen35_cpu_model_t), intent(inout) :: self
    integer(int64), contiguous, intent(in) :: token_ids(:)
    integer(int64), intent(in) :: position
    type(status_t), intent(out) :: stat
    integer :: batch, slot
    type(c_ptr) :: batch_x, batch_residual, batch_normalized, batch_hidden
    type(c_ptr) :: batch_qkv, batch_gate, batch_alpha, batch_beta
    type(c_ptr) :: batch_ffn_gate, batch_ffn_up, batch_q, batch_k, batch_v
    type(c_ptr) :: batch_attention, norm_attn, norm_post

    call stat%clear()
    slot = self%mtp_cuda_slot
    if (slot < 0) return
    batch = size(token_ids)
    if (batch <= 0 .or. batch > 3 .or. &
        .not. c_associated(self%cuda_mtp_verify_hidden) .or. &
        .not. c_associated(self%cuda_mtp_pending_hidden)) then
        call stat%set(FORTAI_INVALID, 'invalid Qwen3.5 MTP verification refresh')
        return
    end if

    ! Mirror llama.cpp's MTP process hook: after target verification,
    ! decode the complete verification batch through the draft head with
    ! target h_nextn shifted right by one row.  Refreshing only accepted
    ! rows leaves draft-generated hidden states in the MTP cache and can
    ! change the very next proposal even when the target stream is exact.
    call select_batch_workspace(self, slot, self%layer_count, batch_x, batch_residual, &
        batch_normalized, batch_hidden, batch_qkv, batch_gate, batch_alpha, batch_beta, &
        batch_ffn_gate, batch_ffn_up, batch_q, batch_k, batch_v, batch_attention, &
        norm_attn, norm_post)
    if (slot == 0) then
        call cuda_qwen35_shift_target_hidden_device(self%cuda, self%cuda_mtp_verify_hidden, &
            self%cuda_mtp_pending_hidden, int(self%hidden_size, c_size_t), batch, batch_residual, stat)
    else
        call cuda_qwen35_shift_target_hidden_device(self%cuda_second, self%cuda_mtp_verify_hidden, &
            self%cuda_mtp_pending_hidden, int(self%hidden_size, c_size_t), batch, batch_residual, stat)
    end if
    if (.not. stat%is_ok()) return
    call qwen35_cpu_mtp_catchup_cuda(self, token_ids, position, stat=stat, &
        target_hidden_device=batch_residual)
end subroutine qwen35_cpu_mtp_refresh_verify_cuda

subroutine qwen35_cpu_mtp_catchup_cuda(self, token_ids, position, target_hidden, stat, &
        target_hidden_device)
    class(qwen35_cpu_model_t), intent(inout) :: self
    integer(int64), contiguous, intent(in) :: token_ids(:)
    integer(int64), intent(in) :: position
    real(real32), contiguous, intent(in), optional :: target_hidden(:)
    type(status_t), intent(out) :: stat
    type(c_ptr), intent(in), optional :: target_hidden_device
    integer(c_size_t) :: hidden, ffn, query, key, value, core, matrix, hidden_bytes
    integer(c_int32_t), allocatable, target :: token_ids_c(:)
    integer :: batch, embed_index, i, slot
    type(c_ptr) :: batch_x, batch_residual, batch_normalized, batch_hidden
    type(c_ptr) :: batch_qkv, batch_gate, batch_alpha, batch_beta
    type(c_ptr) :: batch_ffn_gate, batch_ffn_up, batch_q, batch_k, batch_v
    type(c_ptr) :: batch_attention, norm_attn, norm_post

    call stat%clear()
    batch = size(token_ids)
    if (self%mtp_cuda_slot < 0 .or. batch < 1 .or. position < 0_int64 .or. &
        position + int(batch, int64) > self%max_context .or. &
        size(self%mtp_concat) < batch * self%hidden_size) then
        call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA MTP catch-up batch is invalid')
        return
    end if
    if (present(target_hidden) .eqv. present(target_hidden_device)) then
        call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA MTP catch-up requires one hidden source')
        return
    end if
    if (present(target_hidden)) then
        if (size(target_hidden) /= batch * self%hidden_size) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA MTP host hidden batch is invalid')
            return
        end if
    else if (.not. c_associated(target_hidden_device)) then
        call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA MTP device hidden batch is invalid')
        return
    end if
    if (self%mtp_cuda_slot == 0 .and. batch > self%cuda_batch_capacity) then
        call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA MTP catch-up exceeds primary capacity')
        return
    end if
    if (self%mtp_cuda_slot == 1 .and. batch > self%cuda_batch_capacity_second) then
        call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA MTP catch-up exceeds secondary capacity')
        return
    end if
    allocate(token_ids_c(batch))
    do i = 1, batch
        if (token_ids(i) < 0_int64 .or. token_ids(i) >= self%vocabulary_size) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 CUDA MTP catch-up token is invalid')
            return
        end if
        token_ids_c(i) = int(token_ids(i), c_int32_t)
    end do

    slot = self%mtp_cuda_slot
    hidden = int(self%hidden_size, c_size_t)
    ffn = int(self%feed_forward_size, c_size_t)
    query = int(2 * self%attention_heads * self%attention_head_size, c_size_t)
    key = int(self%attention_heads_kv * self%attention_head_size, c_size_t)
    value = int(self%attention_heads_kv * self%value_length, c_size_t)
    core = int(self%attention_heads * self%value_length, c_size_t)
    matrix = hidden * int(batch, c_size_t)
    hidden_bytes = matrix * int(storage_size(self%x(1)) / 8, c_size_t)
    call select_batch_workspace(self, slot, self%layer_count, batch_x, batch_residual, &
        batch_normalized, batch_hidden, batch_qkv, batch_gate, batch_alpha, batch_beta, &
        batch_ffn_gate, batch_ffn_up, batch_q, batch_k, batch_v, batch_attention, &
        norm_attn, norm_post)
    if (present(target_hidden_device)) batch_residual = target_hidden_device
    embed_index = self%token_embedding
    if (self%mtp_embed_tokens > 0) embed_index = self%mtp_embed_tokens
    if (embed_index == self%token_embedding .and. &
        c_associated(self%cuda_mtp_embedding_weights%handle)) then
        call cuda_q4_embedding_device_batch_slot(self%cuda_q4, slot, &
            self%cuda_mtp_embedding_weights, token_ids_c, batch, batch_x, matrix, stat)
    else if (batch_tensor_ready_on_slot(self, embed_index, slot) .and. &
            self%file%tensors(embed_index)%value_type == GGML_TYPE_Q8_0) then
        if (slot == 0) then
            call cuda_qwen35_embedding_device_batch(self%cuda, self%cuda_weights(embed_index), &
                token_ids_c, batch, batch_x, matrix, stat)
        else
            call cuda_qwen35_embedding_device_batch(self%cuda_second, &
                self%cuda_weights_second(embed_index), token_ids_c, batch, batch_x, matrix, stat)
        end if
    else if (batch_tensor_ready_on_slot(self, embed_index, slot) .and. &
            is_q4_xl_type(self%file%tensors(embed_index)%value_type)) then
        call cuda_q4_embedding_device_batch_slot(self%cuda_q4, slot, &
            self%cuda_q4_weights(embed_index), token_ids_c, batch, batch_x, matrix, stat)
    else
        do i = 1, batch
            call self%file%tensors(embed_index)%get_row(token_ids(i) + 1_int64, &
                self%mtp_concat((i - 1) * self%hidden_size + 1:i * self%hidden_size), stat)
            if (.not. stat%is_ok()) return
        end do
        if (slot == 0) then
            call self%cuda%upload_real(batch_x, self%mtp_concat(:batch * self%hidden_size), stat)
        else
            call self%cuda_second%upload_real(batch_x, self%mtp_concat(:batch * self%hidden_size), stat)
        end if
    end if
    if (stat%is_ok() .and. present(target_hidden)) then
        if (slot == 0) then
            call self%cuda%upload_real(batch_residual, target_hidden, stat)
        else
            call self%cuda_second%upload_real(batch_residual, target_hidden, stat)
        end if
    end if
    if (.not. stat%is_ok()) return

    call batch_rms_norm(self, slot, batch_x, self%cuda_mtp_enorm, batch_normalized, &
        hidden, batch, self%norm_epsilon, stat)
    if (.not. stat%is_ok()) return
    call batch_rms_norm(self, slot, batch_residual, self%cuda_mtp_hnorm, batch_hidden, &
        hidden, batch, self%norm_epsilon, stat)
    if (.not. stat%is_ok()) return
    if (slot == 0) then
        call cuda_qwen35_concat_matrix_device(self%cuda, batch_normalized, batch_hidden, &
            hidden, batch, batch_gate, stat)
    else
        call cuda_qwen35_concat_matrix_device(self%cuda_second, batch_normalized, batch_hidden, &
            hidden, batch, batch_gate, stat)
    end if
    if (.not. stat%is_ok()) return
    call batch_project_one(self, slot, self%mtp_eh_proj, batch_gate, 2_c_size_t * hidden, &
        batch, batch_x, hidden, stat)
    if (.not. stat%is_ok()) return

    call batch_copy(self, slot, batch_x, batch_residual, hidden_bytes, stat)
    if (.not. stat%is_ok()) return
    call batch_rms_norm(self, slot, batch_x, self%cuda_mtp_attn_norm, batch_normalized, &
        hidden, batch, self%norm_epsilon, stat)
    if (.not. stat%is_ok()) return
    call batch_project_triplet(self, slot, self%mtp_layer%attn_q, self%mtp_layer%attn_k, &
        self%mtp_layer%attn_v, batch_normalized, hidden, batch, batch_q, query, batch_k, &
        key, batch_v, value, stat)
    if (.not. stat%is_ok()) return
    call self%mtp_layer%cuda_attention%run_core_device_batch(batch_q, query * int(batch, c_size_t), &
        batch_k, key * int(batch, c_size_t), batch_v, value * int(batch, c_size_t), &
        int(position), batch, batch_attention, core * int(batch, c_size_t), stat)
    if (.not. stat%is_ok()) return
    call batch_project_one(self, slot, self%mtp_layer%attn_out, batch_attention, core, &
        batch, batch_hidden, hidden, stat)
    if (.not. stat%is_ok()) return
    call batch_add(self, slot, batch_hidden, batch_residual, batch_x, matrix, stat)
    if (.not. stat%is_ok()) return

    call batch_copy(self, slot, batch_x, batch_residual, hidden_bytes, stat)
    if (.not. stat%is_ok()) return
    call batch_rms_norm(self, slot, batch_x, self%cuda_mtp_post_norm, batch_normalized, &
        hidden, batch, self%norm_epsilon, stat)
    if (.not. stat%is_ok()) return
    call batch_ffn(self, slot, self%mtp_layer%ffn_gate, self%mtp_layer%ffn_up, self%mtp_layer%ffn_down, &
        batch_normalized, hidden, batch, batch_hidden, hidden, stat)
    if (.not. stat%is_ok()) return
    call batch_add(self, slot, batch_hidden, batch_residual, batch_x, matrix, stat)
    if (.not. stat%is_ok()) return
    self%mtp_last_pair_position = position + int(batch, int64) - 1_int64
    self%mtp_last_pair_token = token_ids(batch)
end subroutine qwen35_cpu_mtp_catchup_cuda

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
    if (self%mtp_cuda_slot >= 0) then
        if (.not. c_associated(self%cuda_mtp_pending_hidden)) then
            call stat%set(FORTAI_INVALID, 'Qwen3.5 native MTP device hidden state is unavailable')
            return
        end if
        call qwen35_cpu_mtp_prepare_inputs_device(self, token_id, stat, &
            hidden_device=self%cuda_mtp_pending_hidden)
        if (.not. stat%is_ok()) return
        call qwen35_cpu_mtp_pair_cuda(self, position, stat, .true.)
        if (.not. stat%is_ok()) return
        self%mtp_last_pair_position = position
        self%mtp_last_pair_token = token_id
        next_token = self%mtp_device_draft_token
    else
        call qwen35_cpu_mtp_pair(self, token_id, position, self%mtp_pending_hidden, stat)
        if (.not. stat%is_ok()) return
        next_token = int(maxloc(self%mtp_logits, dim=1) - 1, int64)
        logit_sum = sum(self%mtp_logits)
    end if
end subroutine qwen35_cpu_mtp_draft_greedy

integer function fast_path_mode()
    character(len=16) :: value
    integer :: length

    ! FortAI production is always native.  The optional ABI path is kept
    ! only for explicit comparison runs; it is never selected implicitly.
    fast_path_mode = 0
    value = ''
    call get_environment_variable('FORTAI_LLAMA_FASTPATH', value, length=length)
    if (length <= 0) return
    if (length > len(value)) length = len(value)
    select case (trim(value(1:length)))
    case ('0', 'native', 'off', 'none')
        fast_path_mode = 0
    case ('1', 'auto')
        fast_path_mode = 0
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
    else if (self%cuda_q4_resident) then
        ! Q4's GGML scheduler has a separate stream.  Complete the Q8
        ! producer before handing its pointer to that backend; the Q4
        ! call synchronizes its own stream before returning.
        if (allocated(self%cuda_q4_weights)) then
            if (c_associated(self%cuda_q4_weights(tensor_index)%handle)) then
                call cuda_q4_matvec_device(self%cuda_q4, self%cuda_q4_weights(tensor_index), device_input, &
                    input_elements, device_output, output_elements, stat)
                return
            end if
        end if
        call stat%set(FORTAI_UNSUPPORTED, 'quantized tensor is not device resident')
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
    if (self%file%tensors(first_index)%value_type == GGML_TYPE_Q8_0 .and. &
        self%file%tensors(second_index)%value_type == GGML_TYPE_Q8_0 .and. &
        (cuda_q8_on_second(self, first_index) .eqv. cuda_q8_on_second(self, second_index))) then
        if (cuda_q8_on_second(self, first_index)) then
            call cuda_q8_matvec_device_f32_pair(self%cuda_second, self%cuda_weights(first_index), &
                self%cuda_weights(second_index), device_input, input_elements, first_output, first_elements, &
                second_output, second_elements, stat)
        else
            call cuda_q8_matvec_device_f32_pair(self%cuda, self%cuda_weights(first_index), &
                self%cuda_weights(second_index), device_input, input_elements, first_output, first_elements, &
                second_output, second_elements, stat)
        end if
        return
    end if
    if (is_q4_xl_type(self%file%tensors(first_index)%value_type) .and. &
        is_q4_xl_type(self%file%tensors(second_index)%value_type) .and. &
        self%cuda_q4_resident .and. self%cuda_q4_group_enabled) then
        if (allocated(self%cuda_q4_weights)) then
            if (c_associated(self%cuda_q4_weights(first_index)%handle) .and. &
                c_associated(self%cuda_q4_weights(second_index)%handle)) then
                call cuda_q4_matvec_device_pair(self%cuda_q4, self%cuda_q4_weights(first_index), &
                    self%cuda_q4_weights(second_index), device_input, input_elements, first_output, first_elements, &
                    second_output, second_elements, stat)
                return
            end if
        end if
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
        self%cuda_q4_group_enabled) then
        if (allocated(self%cuda_q4_weights)) then
            if (c_associated(self%cuda_q4_weights(first_index)%handle) .and. &
                c_associated(self%cuda_q4_weights(second_index)%handle) .and. &
                c_associated(self%cuda_q4_weights(third_index)%handle)) then
                call cuda_q4_matvec_device_triplet(self%cuda_q4, self%cuda_q4_weights(first_index), &
                    self%cuda_q4_weights(second_index), self%cuda_q4_weights(third_index), device_input, &
                    input_elements, first_output, first_elements, second_output, second_elements, third_output, &
                    third_elements, stat)
                return
            end if
        end if
    end if
    call cuda_device_matvec(self, first_index, device_input, input_elements, first_output, first_elements, stat)
    if (.not. stat%is_ok()) return
    call cuda_device_matvec(self, second_index, device_input, input_elements, second_output, second_elements, stat)
    if (.not. stat%is_ok()) return
    call cuda_device_matvec(self, third_index, device_input, input_elements, third_output, third_elements, stat)
end subroutine cuda_device_matvec_triplet

subroutine cuda_device_matvec_quad(self, first_index, second_index, third_index, fourth_index, device_input, &
        input_elements, first_output, first_elements, second_output, second_elements, third_output, &
        third_elements, fourth_output, fourth_elements, stat)
    class(qwen35_cpu_model_t), intent(inout) :: self
    integer, intent(in) :: first_index, second_index, third_index, fourth_index
    type(c_ptr), intent(in) :: device_input, first_output, second_output, third_output, fourth_output
    integer(c_size_t), intent(in) :: input_elements, first_elements, second_elements, third_elements, fourth_elements
    type(status_t), intent(out) :: stat
    logical :: all_q4, ready

    call stat%clear()
    all_q4 = is_q4_xl_type(self%file%tensors(first_index)%value_type) .and. &
        is_q4_xl_type(self%file%tensors(second_index)%value_type) .and. &
        is_q4_xl_type(self%file%tensors(third_index)%value_type) .and. &
        is_q4_xl_type(self%file%tensors(fourth_index)%value_type)
    ready = .false.
    if (all_q4 .and. self%cuda_q4_resident .and. self%cuda_q4_group_enabled .and. &
        allocated(self%cuda_q4_weights)) then
        ready = c_associated(self%cuda_q4_weights(first_index)%handle) .and. &
            c_associated(self%cuda_q4_weights(second_index)%handle) .and. &
            c_associated(self%cuda_q4_weights(third_index)%handle) .and. &
            c_associated(self%cuda_q4_weights(fourth_index)%handle)
    end if
    ! The native Q4 group scheduler can partition one quad across both
    ! resident devices and bridge only the remote outputs.  Keeping the
    ! four projections in one call is important for split Q4_XL: the
    ! previous all-primary guard fell back to two pair calls whenever a
    ! recurrent quad straddled the cards, doubling graph/event setup.
    if (ready) then
        call cuda_q4_matvec_device_quad(self%cuda_q4, self%cuda_q4_weights(first_index), &
            self%cuda_q4_weights(second_index), self%cuda_q4_weights(third_index), &
            self%cuda_q4_weights(fourth_index), device_input, input_elements, first_output, first_elements, &
            second_output, second_elements, third_output, third_elements, fourth_output, fourth_elements, stat)
        return
    end if
    call cuda_device_matvec_pair(self, first_index, second_index, device_input, input_elements, first_output, &
        first_elements, second_output, second_elements, stat)
    if (.not. stat%is_ok()) return
    call cuda_device_matvec_pair(self, third_index, fourth_index, device_input, input_elements, third_output, &
        third_elements, fourth_output, fourth_elements, stat)
end subroutine cuda_device_matvec_quad

subroutine cuda_device_embedding(self, token_id, device_output, output_elements, stat)
    class(qwen35_cpu_model_t), intent(inout) :: self
    integer(int64), intent(in) :: token_id
    type(c_ptr), intent(in) :: device_output
    integer(c_size_t), intent(in) :: output_elements
    type(status_t), intent(out) :: stat
    logical :: q4_embedding_ready

    call stat%clear()
    q4_embedding_ready = .false.
    if (self%cuda_q4_resident) then
        if (allocated(self%cuda_q4_weights)) then
            q4_embedding_ready = c_associated(self%cuda_q4_weights(self%token_embedding)%handle)
        end if
    end if
    if (self%file%tensors(self%token_embedding)%value_type == GGML_TYPE_Q8_0) then
        call cuda_qwen35_embedding_device(self%cuda, self%cuda_weights(self%token_embedding), &
            int(token_id, c_int64_t), device_output, output_elements, stat)
    else if (q4_embedding_ready) then
        call cuda_q4_embedding_device(self%cuda_q4, self%cuda_q4_weights(self%token_embedding), &
            int(token_id, c_int64_t), device_output, output_elements, stat)
    else if (self%cuda_q4_resident) then
        ! MTP-only Q4 sidecars remain host-backed; the main token
        ! embedding takes the resident get-rows path above.
        call self%file%tensors(self%token_embedding)%get_row(token_id + 1_int64, self%x, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%upload_real(device_output, self%x, stat)
    else
        call stat%set(FORTAI_UNSUPPORTED, 'token embedding is not device resident')
    end if
end subroutine cuda_device_embedding

subroutine forward_ffn_device(self, layer_index, stat)
    class(qwen35_cpu_model_t), intent(inout) :: self
    integer, intent(in) :: layer_index
    type(status_t), intent(out) :: stat
    integer(c_size_t) :: hidden_elements, ffn_elements
    character(len=8) :: disable_env
    integer :: disable_length

    call stat%clear()
    hidden_elements = int(size(self%x), c_size_t)
    ffn_elements = int(self%feed_forward_size, c_size_t)
    ! Diagnostic escape hatch: retain the resident layer kernels while
    ! evaluating only the FFN through the established host oracle.  This
    ! is intentionally environment-gated and is removed after kernel
    ! isolation; normal production execution never takes this branch.
    disable_env = ''
    call get_environment_variable('FORTAI_DISABLE_CUDA_FFN_DEVICE', disable_env, length=disable_length)
    if (disable_length > 0 .and. disable_env(1:1) == '1') then
        call self%cuda%download_real(self%cuda_normalized, self%normalized, stat)
        if (.not. stat%is_ok()) return
        call model_matvec_pair(self, self%layers(layer_index)%ffn_gate, self%layers(layer_index)%ffn_up, &
            self%normalized, self%ffn_gate_work, self%ffn_up_work, stat)
        if (.not. stat%is_ok()) return
        call silu_product(self%ffn_gate_work, self%ffn_up_work)
        call model_matvec(self, self%layers(layer_index)%ffn_down, self%ffn_gate_work, self%hidden_work, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%upload_real(self%cuda_hidden, self%hidden_work, stat)
        return
    end if
    if (self%cuda_q4_resident .and. self%cuda_q4_split .and. self%cuda_q4_group_enabled .and. &
        c_associated(self%cuda_second%handle) .and. cuda_q4_on_second(self, self%layers(layer_index)%ffn_gate) .and. &
        cuda_q4_on_second(self, self%layers(layer_index)%ffn_up) .and. &
        cuda_q4_on_second(self, self%layers(layer_index)%ffn_down) .and. &
        c_associated(self%cuda_ffn_gate_device_second)) then
        ! Keep a remote Q4 FFN on the second device.  The specialized
        ! gate/up and down graphs share one scheduler boundary; only the
        ! input and final hidden vector cross the bridge.
        call cuda_q4_matvec_device_swiglu_down(self%cuda_q4, &
            self%cuda_q4_weights(self%layers(layer_index)%ffn_gate), &
            self%cuda_q4_weights(self%layers(layer_index)%ffn_up), &
            self%cuda_q4_weights(self%layers(layer_index)%ffn_down), &
            self%cuda_normalized, hidden_elements, self%cuda_hidden, hidden_elements, stat)
        return
    end if
    if (self%cuda_q4_resident .and. self%cuda_q4_split .and. self%cuda_q4_group_enabled .and. &
        c_associated(self%cuda_second%handle) .and. cuda_q4_on_second(self, self%layers(layer_index)%ffn_gate) .and. &
        cuda_q4_on_second(self, self%layers(layer_index)%ffn_up) .and. &
        cuda_q4_on_second(self, self%layers(layer_index)%ffn_down) .and. &
        c_associated(self%cuda_ffn_gate_device_second) .and. c_associated(self%cuda_ffn_up_device_second)) then
        ! Q4_XL may deliberately use different quantizers for gate and up.
        ! Keep the grouped resident path for that fallback while preserving
        ! the same single activation bridge and remote down projection.
        call cuda_q4_matvec_device_group_remote_output(self%cuda_q4, &
            self%cuda_q4_weights(self%layers(layer_index)%ffn_gate), &
            self%cuda_q4_weights(self%layers(layer_index)%ffn_up), &
            self%cuda_q4_weights(self%layers(layer_index)%ffn_down), 2, self%cuda_normalized, hidden_elements, &
            self%cuda_ffn_gate_device_second, ffn_elements, self%cuda_ffn_up_device_second, ffn_elements, &
            c_null_ptr, 0_c_size_t, stat)
        if (.not. stat%is_ok()) return
        call cuda_qwen35_silu_product_device(self%cuda_second, self%cuda_ffn_gate_device_second, &
            self%cuda_ffn_up_device_second, ffn_elements, stat)
        if (.not. stat%is_ok()) return
        call cuda_q4_matvec_device_remote_input(self%cuda_q4, &
            self%cuda_q4_weights(self%layers(layer_index)%ffn_down), &
            self%cuda_ffn_gate_device_second, ffn_elements, self%cuda_hidden, hidden_elements, stat)
        return
    end if
    if (cuda_ffn_ready(self, self%layers(layer_index))) then
        call cuda_q8_ffn_device(self%cuda, self%cuda_weights(self%layers(layer_index)%ffn_gate), &
            self%cuda_weights(self%layers(layer_index)%ffn_up), &
            self%cuda_weights(self%layers(layer_index)%ffn_down), self%cuda_normalized, &
            hidden_elements, self%cuda_hidden, hidden_elements, stat)
        return
    end if
    if (is_q4_xl_type(self%file%tensors(self%layers(layer_index)%ffn_gate)%value_type) .and. &
        is_q4_xl_type(self%file%tensors(self%layers(layer_index)%ffn_up)%value_type) .and. &
        is_q4_xl_type(self%file%tensors(self%layers(layer_index)%ffn_down)%value_type) .and. &
        self%cuda_q4_resident .and. self%cuda_q4_group_enabled) then
        if (allocated(self%cuda_q4_weights)) then
            if (c_associated(self%cuda_q4_weights(self%layers(layer_index)%ffn_gate)%handle) .and. &
                c_associated(self%cuda_q4_weights(self%layers(layer_index)%ffn_up)%handle) .and. &
                c_associated(self%cuda_q4_weights(self%layers(layer_index)%ffn_down)%handle) .and. &
                (cuda_q4_on_second(self, self%layers(layer_index)%ffn_gate) .eqv. &
                cuda_q4_on_second(self, self%layers(layer_index)%ffn_up)) .and. &
                (cuda_q4_on_second(self, self%layers(layer_index)%ffn_gate) .eqv. &
                cuda_q4_on_second(self, self%layers(layer_index)%ffn_down))) then
                call cuda_q4_matvec_device_swiglu_down(self%cuda_q4, &
                    self%cuda_q4_weights(self%layers(layer_index)%ffn_gate), &
                    self%cuda_q4_weights(self%layers(layer_index)%ffn_up), &
                    self%cuda_q4_weights(self%layers(layer_index)%ffn_down), self%cuda_normalized, &
                    hidden_elements, self%cuda_hidden, hidden_elements, stat)
                return
            end if
        end if
    end if
    if (is_q4_xl_type(self%file%tensors(self%layers(layer_index)%ffn_gate)%value_type) .and. &
        is_q4_xl_type(self%file%tensors(self%layers(layer_index)%ffn_up)%value_type) .and. &
        self%cuda_q4_resident .and. self%cuda_q4_group_enabled) then
        if (allocated(self%cuda_q4_weights)) then
            if (c_associated(self%cuda_q4_weights(self%layers(layer_index)%ffn_gate)%handle) .and. &
                c_associated(self%cuda_q4_weights(self%layers(layer_index)%ffn_up)%handle) .and. &
                (cuda_q4_on_second(self, self%layers(layer_index)%ffn_gate) .eqv. &
                cuda_q4_on_second(self, self%layers(layer_index)%ffn_up))) then
                call cuda_q4_matvec_device_swiglu(self%cuda_q4, &
                    self%cuda_q4_weights(self%layers(layer_index)%ffn_gate), &
                    self%cuda_q4_weights(self%layers(layer_index)%ffn_up), self%cuda_normalized, hidden_elements, &
                    self%cuda_ffn_gate_device, ffn_elements, stat)
                if (.not. stat%is_ok()) return
            else
                call cuda_device_matvec_pair(self, self%layers(layer_index)%ffn_gate, &
                    self%layers(layer_index)%ffn_up, self%cuda_normalized, hidden_elements, &
                    self%cuda_ffn_gate_device, ffn_elements, self%cuda_ffn_up_device, ffn_elements, stat)
                if (.not. stat%is_ok()) return
                call cuda_qwen35_silu_product_device(self%cuda, self%cuda_ffn_gate_device, &
                    self%cuda_ffn_up_device, ffn_elements, stat)
                if (.not. stat%is_ok()) return
            end if
        else
            call cuda_device_matvec_pair(self, self%layers(layer_index)%ffn_gate, &
                self%layers(layer_index)%ffn_up, self%cuda_normalized, hidden_elements, &
                self%cuda_ffn_gate_device, ffn_elements, self%cuda_ffn_up_device, ffn_elements, stat)
            if (.not. stat%is_ok()) return
            call cuda_qwen35_silu_product_device(self%cuda, self%cuda_ffn_gate_device, &
                self%cuda_ffn_up_device, ffn_elements, stat)
            if (.not. stat%is_ok()) return
        end if
    else
        call cuda_device_matvec_pair(self, self%layers(layer_index)%ffn_gate, self%layers(layer_index)%ffn_up, &
            self%cuda_normalized, hidden_elements, self%cuda_ffn_gate_device, ffn_elements, &
            self%cuda_ffn_up_device, ffn_elements, stat)
        if (.not. stat%is_ok()) return
        call cuda_qwen35_silu_product_device(self%cuda, self%cuda_ffn_gate_device, &
            self%cuda_ffn_up_device, ffn_elements, stat)
        if (.not. stat%is_ok()) return
    end if
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
    if (cuda_norm_device_disabled()) then
        call self%cuda%download_real(self%cuda_x, self%x, stat)
        if (.not. stat%is_ok()) return
        call rms_norm(self%x, self%file%tensors(self%layers(layer_index)%attn_norm), &
            self%norm_epsilon, self%normalized, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%upload_real(self%cuda_normalized, self%normalized, stat)
    else
        call cuda_qwen35_rms_norm_device(self%cuda, self%cuda_x, &
            self%cuda_attn_norm(layer_index), self%cuda_normalized, hidden_elements, &
            self%norm_epsilon, stat)
    end if
    if (.not. stat%is_ok()) return
    if (all_q8_recurrent_weights(self, layer_index)) then
        call self%layers(layer_index)%cuda_recurrent%run_device(self%cuda_normalized, hidden_elements, &
            self%cuda_hidden, hidden_elements, stat)
    else
        call cuda_device_matvec_quad(self, self%layers(layer_index)%attn_qkv, &
            self%layers(layer_index)%attn_gate, self%layers(layer_index)%ssm_alpha, &
            self%layers(layer_index)%ssm_beta, self%cuda_normalized, hidden_elements, self%cuda_qkv_device, &
            int(self%recurrent_conv_size, c_size_t), self%cuda_gate_device, &
            int(self%recurrent_inner_size, c_size_t), self%cuda_alpha_device, &
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
    if (cuda_norm_device_disabled()) then
        call self%cuda%download_real(self%cuda_x, self%x, stat)
        if (.not. stat%is_ok()) return
        call rms_norm(self%x, self%file%tensors(self%layers(layer_index)%post_norm), &
            self%norm_epsilon, self%normalized, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%upload_real(self%cuda_normalized, self%normalized, stat)
    else
        call cuda_qwen35_rms_norm_device(self%cuda, self%cuda_x, &
            self%cuda_post_norm(layer_index), self%cuda_normalized, hidden_elements, &
            self%norm_epsilon, stat)
    end if
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

subroutine forward_output_device_greedy(self, greedy_token, stat)
    class(qwen35_cpu_model_t), intent(inout) :: self
    integer(int64), intent(out) :: greedy_token
    type(status_t), intent(out) :: stat
    integer(c_size_t) :: hidden_elements, logits_elements
    logical :: remote_output

    call stat%clear()
    greedy_token = -1_int64
    hidden_elements = int(size(self%x), c_size_t)
    logits_elements = int(size(self%logits), c_size_t)
    call cuda_qwen35_rms_norm_device(self%cuda, self%cuda_x, self%cuda_output_norm, &
        self%cuda_normalized, hidden_elements, self%norm_epsilon, stat)
    if (.not. stat%is_ok()) return

    remote_output = .false.
    if (self%cuda_q4_resident .and. self%cuda_q4_split .and. c_associated(self%cuda_second%handle) .and. &
        allocated(self%cuda_q4_weights) .and. allocated(self%cuda_q4_weight_device)) then
        if (self%output > 0 .and. self%output <= size(self%file%tensors) .and. &
            self%output <= size(self%cuda_q4_weights) .and. self%output <= size(self%cuda_q4_weight_device)) then
            if (is_q4_xl_type(self%file%tensors(self%output)%value_type) .and. &
                cuda_q4_on_second(self, self%output) .and. c_associated(self%cuda_q4_weights(self%output)%handle) .and. &
                c_associated(self%cuda_logits_second)) remote_output = .true.
        end if
    end if
    if (remote_output) then
        ! The Q4 projection and argmax stay on the second scheduler/device,
        ! exactly like llama.cpp's ggml_argmax backend node.  Reusing the
        ! existing count-one group API keeps one validated bridge and one
        ! cached GGML-CUDA plan without introducing a wrapper runtime.
        call cuda_q4_matvec_device_group_remote_output(self%cuda_q4, &
            self%cuda_q4_weights(self%output), self%cuda_q4_weights(self%output), &
            self%cuda_q4_weights(self%output), 1, self%cuda_normalized, hidden_elements, &
            self%cuda_logits_second, logits_elements, c_null_ptr, 0_c_size_t, c_null_ptr, 0_c_size_t, stat)
        if (.not. stat%is_ok()) return
        block
            integer :: index
            call self%cuda_second%argmax_device(self%cuda_logits_second, &
                int(self%vocabulary_size, c_size_t), index, stat)
            if (.not. stat%is_ok()) return
            greedy_token = int(index, int64)
        end block
        return
    end if
    call cuda_device_matvec(self, self%output, self%cuda_normalized, hidden_elements, &
        self%cuda_logits, logits_elements, stat)
end subroutine forward_output_device_greedy

subroutine forward_attention_device(self, layer_index, position, stat)
    class(qwen35_cpu_model_t), intent(inout) :: self
    integer, intent(in) :: layer_index
    integer(int64), intent(in) :: position
    type(status_t), intent(out) :: stat
    integer(c_size_t) :: hidden_elements, hidden_bytes
    logical :: on_second

    call stat%clear()
    hidden_elements = int(size(self%x), c_size_t)
    hidden_bytes = hidden_elements * int(storage_size(self%x(1)) / 8, c_size_t)
    call cuda_qwen35_copy_device(self%cuda, self%cuda_x, self%cuda_residual, hidden_bytes, stat)
    if (.not. stat%is_ok()) return
    if (cuda_norm_device_disabled()) then
        call self%cuda%download_real(self%cuda_x, self%x, stat)
        if (.not. stat%is_ok()) return
        call rms_norm(self%x, self%file%tensors(self%layers(layer_index)%attn_norm), &
            self%norm_epsilon, self%normalized, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%upload_real(self%cuda_normalized, self%normalized, stat)
    else
        call cuda_qwen35_rms_norm_device(self%cuda, self%cuda_x, &
            self%cuda_attn_norm(layer_index), self%cuda_normalized, hidden_elements, &
            self%norm_epsilon, stat)
    end if
    if (.not. stat%is_ok()) return
    on_second = cuda_attention_on_second(self, layer_index)
    if (all_q8_attention_weights(self, layer_index)) then
        call self%layers(layer_index)%cuda_attention%run_device(self%cuda_normalized, hidden_elements, &
            int(position), self%cuda_hidden, hidden_elements, stat)
    else
        if (on_second .and. self%cuda_q4_group_enabled .and. &
            cuda_q4_triplet_on_second(self, self%layers(layer_index)%attn_q, &
            self%layers(layer_index)%attn_k, self%layers(layer_index)%attn_v)) then
            call cuda_q4_matvec_device_group_remote_output(self%cuda_q4, &
                self%cuda_q4_weights(self%layers(layer_index)%attn_q), &
                self%cuda_q4_weights(self%layers(layer_index)%attn_k), &
                self%cuda_q4_weights(self%layers(layer_index)%attn_v), 3, self%cuda_normalized, hidden_elements, &
                self%cuda_attention_q_device_second, &
                int(2 * self%attention_heads * self%attention_head_size, c_size_t), &
                self%cuda_attention_k_device_second, int(self%attention_heads_kv * self%attention_head_size, c_size_t), &
                self%cuda_attention_v_device_second, int(self%attention_heads_kv * self%value_length, c_size_t), stat)
        else
            call cuda_device_matvec_triplet(self, self%layers(layer_index)%attn_q, &
                self%layers(layer_index)%attn_k, self%layers(layer_index)%attn_v, self%cuda_normalized, hidden_elements, &
                self%cuda_attention_q_device, int(2 * self%attention_heads * self%attention_head_size, c_size_t), &
                self%cuda_attention_k_device, int(self%attention_heads_kv * self%attention_head_size, c_size_t), &
                self%cuda_attention_v_device, int(self%attention_heads_kv * self%value_length, c_size_t), stat)
        end if
        if (.not. stat%is_ok()) return
        if (on_second) then
            if (.not. (self%cuda_q4_group_enabled .and. cuda_q4_triplet_on_second(self, self%layers(layer_index)%attn_q, &
                self%layers(layer_index)%attn_k, self%layers(layer_index)%attn_v))) then
                call cuda_attention_move_qkv_to_second(self, stat)
                if (.not. stat%is_ok()) return
            end if
            call self%cuda_second%set_position(int(position), stat)
            if (.not. stat%is_ok()) return
            call self%layers(layer_index)%cuda_attention%run_core_device(self%cuda_attention_q_device_second, &
                int(2 * self%attention_heads * self%attention_head_size, c_size_t), &
                self%cuda_attention_k_device_second, &
                int(self%attention_heads_kv * self%attention_head_size, c_size_t), &
                self%cuda_attention_v_device_second, int(self%attention_heads_kv * self%value_length, c_size_t), &
                int(position), self%cuda_attention_work_device_second, &
                int(self%attention_heads * self%value_length, c_size_t), stat)
            if (.not. stat%is_ok()) return
            if (.not. (self%cuda_q4_group_enabled .and. cuda_q4_triplet_on_second(self, &
                self%layers(layer_index)%attn_q, self%layers(layer_index)%attn_k, &
                self%layers(layer_index)%attn_v) .and. cuda_q4_on_second(self, &
                self%layers(layer_index)%attn_out))) then
                call self%cuda_second%download_real(self%cuda_attention_work_device_second, self%attention_work, stat)
                if (.not. stat%is_ok()) return
                call self%cuda%upload_real(self%cuda_attention_work_device, self%attention_work, stat)
                if (.not. stat%is_ok()) return
            end if
        else
            call self%layers(layer_index)%cuda_attention%run_core_device(self%cuda_attention_q_device, &
                int(2 * self%attention_heads * self%attention_head_size, c_size_t), self%cuda_attention_k_device, &
                int(self%attention_heads_kv * self%attention_head_size, c_size_t), self%cuda_attention_v_device, &
                int(self%attention_heads_kv * self%value_length, c_size_t), int(position), &
                self%cuda_attention_work_device, int(self%attention_heads * self%value_length, c_size_t), stat)
            if (.not. stat%is_ok()) return
        end if
        if (on_second .and. cuda_q4_on_second(self, self%layers(layer_index)%attn_out)) then
            call cuda_q4_matvec_device_remote_input(self%cuda_q4, &
                self%cuda_q4_weights(self%layers(layer_index)%attn_out), self%cuda_attention_work_device_second, &
                int(self%attention_heads * self%value_length, c_size_t), self%cuda_hidden, hidden_elements, stat)
        else
            call cuda_device_matvec(self, self%layers(layer_index)%attn_out, self%cuda_attention_work_device, &
                int(self%attention_heads * self%value_length, c_size_t), self%cuda_hidden, hidden_elements, stat)
        end if
    end if
    if (.not. stat%is_ok()) return
    call cuda_qwen35_add_device(self%cuda, self%cuda_hidden, self%cuda_residual, self%cuda_x, &
        hidden_elements, stat)
    if (.not. stat%is_ok()) return
    call cuda_qwen35_copy_device(self%cuda, self%cuda_x, self%cuda_residual, hidden_bytes, stat)
    if (.not. stat%is_ok()) return
    if (cuda_norm_device_disabled()) then
        call self%cuda%download_real(self%cuda_x, self%x, stat)
        if (.not. stat%is_ok()) return
        call rms_norm(self%x, self%file%tensors(self%layers(layer_index)%post_norm), &
            self%norm_epsilon, self%normalized, stat)
        if (.not. stat%is_ok()) return
        call self%cuda%upload_real(self%cuda_normalized, self%normalized, stat)
    else
        call cuda_qwen35_rms_norm_device(self%cuda, self%cuda_x, &
            self%cuda_post_norm(layer_index), self%cuda_normalized, hidden_elements, &
            self%norm_epsilon, stat)
    end if
    if (.not. stat%is_ok()) return
    call forward_ffn_device(self, layer_index, stat)
    if (.not. stat%is_ok()) return
    call cuda_qwen35_add_device(self%cuda, self%cuda_hidden, self%cuda_residual, self%cuda_x, &
        hidden_elements, stat)
end subroutine forward_attention_device

subroutine cuda_attention_move_qkv_to_second(self, stat)
    class(qwen35_cpu_model_t), intent(inout) :: self
    type(status_t), intent(out) :: stat
    integer :: query_elements, key_elements, value_elements, offset

    call stat%clear()
    query_elements = 2 * self%attention_heads * self%attention_head_size
    key_elements = self%attention_heads_kv * self%attention_head_size
    value_elements = self%attention_heads_kv * self%value_length
    if (size(self%qkv_download_work) < query_elements + key_elements + value_elements) then
        call stat%set(FORTAI_INVALID, 'CUDA split attention transfer workspace is too small')
        return
    end if
    offset = 1
    call self%cuda%download_real(self%cuda_attention_q_device, &
        self%qkv_download_work(offset:offset + query_elements - 1), stat)
    if (.not. stat%is_ok()) return
    call self%cuda_second%upload_real(self%cuda_attention_q_device_second, &
        self%qkv_download_work(offset:offset + query_elements - 1), stat)
    if (.not. stat%is_ok()) return
    offset = offset + query_elements
    call self%cuda%download_real(self%cuda_attention_k_device, &
        self%qkv_download_work(offset:offset + key_elements - 1), stat)
    if (.not. stat%is_ok()) return
    call self%cuda_second%upload_real(self%cuda_attention_k_device_second, &
        self%qkv_download_work(offset:offset + key_elements - 1), stat)
    if (.not. stat%is_ok()) return
    offset = offset + key_elements
    call self%cuda%download_real(self%cuda_attention_v_device, &
        self%qkv_download_work(offset:offset + value_elements - 1), stat)
    if (.not. stat%is_ok()) return
    call self%cuda_second%upload_real(self%cuda_attention_v_device_second, &
        self%qkv_download_work(offset:offset + value_elements - 1), stat)
end subroutine cuda_attention_move_qkv_to_second

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

integer function q4_layer_for_tensor(self, tensor_index)
    class(qwen35_cpu_model_t), intent(in) :: self
    integer, intent(in) :: tensor_index
    integer :: layer_number, j
    integer :: indices(20)

    q4_layer_for_tensor = 0
    if (tensor_index <= 0 .or. tensor_index > size(self%file%tensors)) return
    if (.not. allocated(self%layers)) return
    do layer_number = 1, size(self%layers)
        indices = [self%layers(layer_number)%attn_norm, self%layers(layer_number)%post_norm, &
            self%layers(layer_number)%ffn_gate, self%layers(layer_number)%ffn_up, &
            self%layers(layer_number)%ffn_down, self%layers(layer_number)%attn_qkv, &
            self%layers(layer_number)%attn_gate, self%layers(layer_number)%attn_q, &
            self%layers(layer_number)%attn_k, self%layers(layer_number)%attn_v, &
            self%layers(layer_number)%attn_out, self%layers(layer_number)%q_norm, &
            self%layers(layer_number)%k_norm, self%layers(layer_number)%ssm_a, &
            self%layers(layer_number)%ssm_alpha, self%layers(layer_number)%ssm_beta, &
            self%layers(layer_number)%ssm_conv, self%layers(layer_number)%ssm_dt, &
            self%layers(layer_number)%ssm_norm, self%layers(layer_number)%ssm_out]
        do j = 1, size(indices)
            if (indices(j) == tensor_index) then
                q4_layer_for_tensor = layer_number
                return
            end if
        end do
    end do
    if (self%mtp_available) then
        indices = [self%mtp_layer%attn_norm, self%mtp_layer%post_norm, self%mtp_layer%ffn_gate, &
            self%mtp_layer%ffn_up, self%mtp_layer%ffn_down, self%mtp_layer%attn_qkv, &
            self%mtp_layer%attn_gate, self%mtp_layer%attn_q, self%mtp_layer%attn_k, &
            self%mtp_layer%attn_v, self%mtp_layer%attn_out, self%mtp_layer%q_norm, &
            self%mtp_layer%k_norm, self%mtp_layer%ssm_a, self%mtp_layer%ssm_alpha, &
            self%mtp_layer%ssm_beta, self%mtp_layer%ssm_conv, self%mtp_layer%ssm_dt, &
            self%mtp_layer%ssm_norm, self%mtp_layer%ssm_out]
        do j = 1, size(indices)
            if (indices(j) == tensor_index) then
                q4_layer_for_tensor = size(self%layers) + 1
                return
            end if
        end do
        ! The input fusion and optional shared output projection are
        ! part of the MTP block even though their GGUF names use the
        ! `nextn` namespace instead of the ordinary block namespace.
        ! Keep them with the MTP layer so its decode graph has one CUDA
        ! owner and never submits work through a foreign-device stream.
        if (tensor_index == self%mtp_eh_proj .or. &
            tensor_index == self%mtp_embed_tokens .or. &
            tensor_index == self%mtp_shared_head_head) then
            q4_layer_for_tensor = size(self%layers) + 1
            return
        end if
    end if
end function q4_layer_for_tensor

logical function q4_recurrent_projection_tensor(self, tensor_index)
    class(qwen35_cpu_model_t), intent(in) :: self
    integer, intent(in) :: tensor_index
    integer :: layer_number

    q4_recurrent_projection_tensor = .false.
    if (.not. allocated(self%layers)) return
    do layer_number = 1, size(self%layers)
        if (.not. self%layers(layer_number)%recurrent) cycle
        if (tensor_index == self%layers(layer_number)%attn_qkv .or. &
            tensor_index == self%layers(layer_number)%attn_gate) then
            q4_recurrent_projection_tensor = .true.
            return
        end if
    end do
end function q4_recurrent_projection_tensor


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

logical function cuda_attention_split_requested(self, layer_index)
    class(qwen35_cpu_model_t), intent(in) :: self
    integer, intent(in) :: layer_index
    integer :: indices(4), i
    logical :: all_q4, split_attention_enabled
    character(len=16) :: split_attention_value
    character(len=128) :: tensor_split_value
    integer :: split_attention_length, tensor_split_length, tensor_split_status
    real(real64) :: tensor_split_fraction(2), tensor_split_sum, layer_position

    cuda_attention_split_requested = .false.
    if (.not. self%cuda_q4_split .or. .not. allocated(self%cuda_q4_weight_device)) return
    split_attention_value = ''
    call get_environment_variable('FORTAI_CUDA_Q4_SPLIT_ATTENTION', split_attention_value, &
        length=split_attention_length)
    split_attention_enabled = cuda_split_mode_layer()
    if (split_attention_length > 0) then
        select case (trim(split_attention_value(:min(split_attention_length, len(split_attention_value)))))
        case ('0', 'false', 'off', 'no')
            split_attention_enabled = .false.
        case ('1', 'true', 'on', 'yes')
            split_attention_enabled = .true.
        case default
            split_attention_enabled = .false.
        end select
    end if
    ! Layer placement is also the default for llama.cpp.  Keep the full
    ! attention/KV state on the same side of the normalized split as its
    ! Q4 projections; an explicit FORTAI_CUDA_Q4_SPLIT_ATTENTION=0 still
    ! provides the diagnostic single-device fallback.
    if (.not. split_attention_enabled) return
    if (layer_index <= 0 .or. layer_index > size(self%layers)) return
    if (self%layers(layer_index)%recurrent) return
    indices = [self%layers(layer_index)%attn_q, self%layers(layer_index)%attn_k, &
        self%layers(layer_index)%attn_v, self%layers(layer_index)%attn_out]
    all_q4 = .true.
    do i = 1, size(indices)
        if (indices(i) <= 0 .or. indices(i) > size(self%file%tensors)) return
        if (.not. is_q4_xl_type(self%file%tensors(indices(i))%value_type)) all_q4 = .false.
    end do
    ! At short contexts the primary board has ample KV headroom and
    ! keeping mixed Q4/Q8 projections together avoids an unnecessary
    ! QKV/workspace bridge.  At production long context the KV state is
    ! the dominant allocation; allow mixed blocks to move their state to
    ! the remote board so both cards remain within their VRAM budgets.
    if (.not. all_q4 .and. self%max_context <= 32768_int64) return
    tensor_split_fraction = 0.5_real64
    tensor_split_value = ''
    call get_environment_variable('FORTAI_TENSOR_SPLIT', tensor_split_value, length=tensor_split_length)
    if (tensor_split_length <= 0) call get_environment_variable('LLAMA_ARG_TENSOR_SPLIT', tensor_split_value, &
        length=tensor_split_length)
    if (tensor_split_length > 0 .and. tensor_split_length <= len(tensor_split_value)) then
        tensor_split_status = 0
        read(tensor_split_value(:tensor_split_length), *, iostat=tensor_split_status) tensor_split_fraction
        tensor_split_sum = sum(tensor_split_fraction)
        if (tensor_split_status == 0 .and. all(tensor_split_fraction >= 0.0_real64) .and. &
            tensor_split_sum > 0.0_real64) tensor_split_fraction = tensor_split_fraction / tensor_split_sum
    end if
    layer_position = real(layer_index - 1, real64) / real(max(1, int(self%layer_count) + 1), real64)
    cuda_attention_split_requested = layer_position >= tensor_split_fraction(1) .and. &
        c_associated(self%cuda_second%handle)
end function cuda_attention_split_requested

logical function cuda_attention_on_second(self, layer_index)
    class(qwen35_cpu_model_t), intent(in) :: self
    integer, intent(in) :: layer_index

    cuda_attention_on_second = .false.
    if (.not. allocated(self%cuda_attention_on_second_layer)) return
    if (layer_index <= 0 .or. layer_index > size(self%cuda_attention_on_second_layer)) return
    cuda_attention_on_second = self%cuda_attention_on_second_layer(layer_index)
end function cuda_attention_on_second

logical function q4_attention_tensor_mixed(self, tensor_index)
    class(qwen35_cpu_model_t), intent(in) :: self
    integer, intent(in) :: tensor_index
    integer :: layer_index, i
    integer :: indices(4)
    logical :: all_q4

    q4_attention_tensor_mixed = .false.
    if (.not. allocated(self%layers)) return
    do layer_index = 1, size(self%layers)
        if (self%layers(layer_index)%recurrent) cycle
        indices = [self%layers(layer_index)%attn_q, self%layers(layer_index)%attn_k, &
            self%layers(layer_index)%attn_v, self%layers(layer_index)%attn_out]
        if (.not. any(indices == tensor_index)) cycle
        all_q4 = .true.
        do i = 1, size(indices)
            if (indices(i) <= 0) then
                all_q4 = .false.
                exit
            end if
            if (.not. is_q4_xl_type(self%file%tensors(indices(i))%value_type)) then
                all_q4 = .false.
                exit
            end if
        end do
        q4_attention_tensor_mixed = .not. all_q4
        return
    end do
end function q4_attention_tensor_mixed

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

logical function cuda_norm_device_disabled()
    character(len=8) :: value
    integer :: length

    cuda_norm_device_disabled = .false.
    value = ''
    call get_environment_variable('FORTAI_DISABLE_CUDA_NORM_DEVICE', value, length=length)
    if (length > 0) cuda_norm_device_disabled = value(1:min(length, len(value))) == '1'
end function cuda_norm_device_disabled

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

logical function cuda_layer_kernel_ready(self, layer_index)
    class(qwen35_cpu_model_t), intent(in) :: self
    integer, intent(in) :: layer_index

    cuda_layer_kernel_ready = .false.
    if (.not. allocated(self%layers)) return
    if (layer_index <= 0 .or. layer_index > size(self%layers)) return
    if (self%layers(layer_index)%recurrent) then
        if (.not. c_associated(self%layers(layer_index)%cuda_recurrent%handle)) return
    else
        if (.not. c_associated(self%layers(layer_index)%cuda_attention%handle)) return
    end if
    if (.not. cuda_ffn_device_ready(self, self%layers(layer_index))) return
    cuda_layer_kernel_ready = .true.
end function cuda_layer_kernel_ready

logical function cuda_layer_device_ready(self, layer_index)
    class(qwen35_cpu_model_t), intent(in) :: self
    integer, intent(in) :: layer_index
    character(len=16) :: limit_value, skip_value
    integer :: limit_length, limit, ios, skip_length

    cuda_layer_device_ready = .false.
    limit_value = ''
    call get_environment_variable('FORTAI_CUDA_DEVICE_LAYER_LIMIT', limit_value, length=limit_length)
    if (limit_length > 0) then
        read(limit_value(:min(limit_length, len(limit_value))), *, iostat=ios) limit
        if (ios == 0 .and. limit >= 0 .and. layer_index > limit) return
    end if
    skip_value = ''
    call get_environment_variable('FORTAI_DISABLE_CUDA_RECURRENT_DEVICE', skip_value, length=skip_length)
    if (skip_length > 0 .and. skip_value(1:1) == '1' .and. self%layers(layer_index)%recurrent) return
    skip_value = ''
    call get_environment_variable('FORTAI_DISABLE_CUDA_ATTENTION_DEVICE', skip_value, length=skip_length)
    if (skip_length > 0 .and. skip_value(1:1) == '1' .and. .not. self%layers(layer_index)%recurrent) return
    if (.not. cuda_layer_kernel_ready(self, layer_index)) return
    if (.not. allocated(self%cuda_attn_norm) .or. .not. allocated(self%cuda_post_norm)) return
    if (layer_index > size(self%cuda_attn_norm) .or. layer_index > size(self%cuda_post_norm)) return
    if (.not. c_associated(self%cuda_attn_norm(layer_index))) return
    if (.not. c_associated(self%cuda_post_norm(layer_index))) return
    cuda_layer_device_ready = .true.
end function cuda_layer_device_ready

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

logical function cuda_q4_on_second(self, tensor_index)
    class(qwen35_cpu_model_t), intent(in) :: self
    integer, intent(in) :: tensor_index

    cuda_q4_on_second = .false.
    if (.not. allocated(self%cuda_q4_weight_device)) return
    if (tensor_index <= 0) return
    if (tensor_index > size(self%cuda_q4_weight_device)) return
    cuda_q4_on_second = self%cuda_q4_weight_device(tensor_index) == 2
end function cuda_q4_on_second

logical function cuda_q4_triplet_on_second(self, first_index, second_index, third_index)
    class(qwen35_cpu_model_t), intent(in) :: self
    integer, intent(in) :: first_index, second_index, third_index

    cuda_q4_triplet_on_second = .false.
    if (.not. cuda_q4_on_second(self, first_index)) return
    if (.not. cuda_q4_on_second(self, second_index)) return
    if (.not. cuda_q4_on_second(self, third_index)) return
    cuda_q4_triplet_on_second = .true.
end function cuda_q4_triplet_on_second

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
        (cuda_q8_on_second(self, first_index) .eqv. cuda_q8_on_second(self, second_index)) .and. &
        (cuda_q8_on_second(self, first_index) .eqv. cuda_q8_on_second(self, third_index))) then
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
    if ((cuda_q8_on_second(self, layer%ffn_gate) .eqv. cuda_q8_on_second(self, layer%ffn_up)) .and. &
        (cuda_q8_on_second(self, layer%ffn_gate) .eqv. cuda_q8_on_second(self, layer%ffn_down))) then
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

    call allocate_attention_cache(self, layer)
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
