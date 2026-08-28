module fortai_backend_cuda
    use, intrinsic :: iso_c_binding, only: c_float, c_int, c_int8_t, c_int32_t, c_int64_t, c_loc, c_ptr, &
        c_null_ptr, c_size_t, c_associated
    use fortai_status, only: FORTAI_INVALID, FORTAI_UNSUPPORTED, status_t
    implicit none
    private

    integer(c_int), parameter, public :: FORTAI_CUDA_OK = 0_c_int

    type, public :: cuda_q8_context_t
        type(c_ptr) :: handle = c_null_ptr
    contains
        procedure :: create => cuda_q8_context_create
        procedure :: destroy => cuda_q8_context_destroy
        procedure :: set_position => cuda_q8_context_set_position
        procedure :: synchronize => cuda_q8_context_synchronize
        procedure :: stream => cuda_q8_context_stream
        procedure :: adopt_stream => cuda_q8_context_adopt_stream
        procedure :: capture_begin => cuda_q8_context_capture_begin
        procedure :: capture_end => cuda_q8_context_capture_end
        procedure :: graph_launch => cuda_q8_context_graph_launch
        procedure :: allocate_buffer => cuda_q8_allocate_buffer
        procedure :: free_buffer => cuda_q8_free_buffer
        procedure :: upload => cuda_q8_upload
        procedure :: upload_real => cuda_q8_upload_real
        procedure :: download => cuda_q8_download
        procedure :: download_real => cuda_q8_download_real
        procedure :: argmax_device => cuda_qwen35_argmax_device
        procedure :: last_error => cuda_q8_last_error
    end type cuda_q8_context_t

    type, public :: cuda_q8_weights_t
        type(c_ptr) :: handle = c_null_ptr
    contains
        procedure :: upload => cuda_q8_weights_upload
        procedure :: destroy => cuda_q8_weights_destroy
    end type cuda_q8_weights_t

    type, public :: cuda_q4_context_t
        type(c_ptr) :: handle = c_null_ptr
    contains
        procedure :: create => cuda_q4_context_create
        procedure :: destroy => cuda_q4_context_destroy
        procedure :: synchronize => cuda_q4_context_synchronize
        procedure :: stream => cuda_q4_context_stream
        procedure :: set_consumer_stream => cuda_q4_context_set_consumer_stream
        procedure :: matvec_device => cuda_q4_matvec_device
    end type cuda_q4_context_t

    type, public :: cuda_q4_weights_t
        type(c_ptr) :: handle = c_null_ptr
    contains
        procedure :: upload => cuda_q4_weights_upload
        procedure :: destroy => cuda_q4_weights_destroy
    end type cuda_q4_weights_t

    type, public :: cuda_qwen35_recurrent_t
        type(c_ptr) :: handle = c_null_ptr
    contains
        procedure :: create => cuda_qwen35_recurrent_create
        procedure :: create_state => cuda_qwen35_recurrent_create_state
        procedure :: destroy => cuda_qwen35_recurrent_destroy
        procedure :: reset => cuda_qwen35_recurrent_reset
        procedure :: run => cuda_qwen35_recurrent_run
        procedure :: run_device => cuda_qwen35_recurrent_run_device
        procedure :: run_core_device => cuda_qwen35_recurrent_run_core_device
        procedure :: run_core_device_batch => cuda_qwen35_recurrent_run_core_device_batch
    end type cuda_qwen35_recurrent_t

    type, public :: cuda_qwen35_attention_t
        type(c_ptr) :: handle = c_null_ptr
    contains
        procedure :: create => cuda_qwen35_attention_create
        procedure :: create_state => cuda_qwen35_attention_create_state
        procedure :: destroy => cuda_qwen35_attention_destroy
        procedure :: reset => cuda_qwen35_attention_reset
        procedure :: run_device => cuda_qwen35_attention_run_device
        procedure :: run_core_device => cuda_qwen35_attention_run_core_device
        procedure :: run_core_device_batch => cuda_qwen35_attention_run_core_device_batch
    end type cuda_qwen35_attention_t

    public :: cuda_q8_matvec_host
    public :: cuda_q8_matvec_host_pair
    public :: cuda_q8_matvec_host_triplet
    public :: cuda_q8_matvec_host_triplet_contiguous
    public :: cuda_q8_ffn_host
    public :: cuda_q8_ffn_device
    public :: cuda_qwen35_copy_device
    public :: cuda_qwen35_add_device
    public :: cuda_qwen35_add_matrix_device
    public :: cuda_qwen35_copy_column_device
    public :: cuda_qwen35_rms_norm_device
    public :: cuda_qwen35_rms_norm_matrix_device
    public :: cuda_qwen35_silu_product_device
    public :: cuda_qwen35_silu_product_matrix_device
    public :: cuda_qwen35_argmax_device
    public :: cuda_q8_matvec_resident
    public :: cuda_q8_matvec_device_f32
    public :: cuda_q8_matvec_device_f32_pair
    public :: cuda_q8_reserve_matvec_scratch
    public :: cuda_q8_matmul_device_f32
    public :: cuda_qwen35_embedding_device
    public :: cuda_qwen35_embedding_device_batch
    public :: cuda_q4_matvec_host
    public :: cuda_q4_matvec_host_pair
    public :: cuda_q4_matvec_host_triplet
    public :: cuda_q4_matvec_device
    public :: cuda_q4_matvec_device_swiglu
    public :: cuda_q4_matvec_device_swiglu_remote_output
    public :: cuda_q4_matvec_device_swiglu_down
    public :: cuda_q4_matvec_device_pair
    public :: cuda_q4_matvec_device_triplet
    public :: cuda_q4_matvec_device_quad
    public :: cuda_q4_matmul_device_pair
    public :: cuda_q4_matmul_device_triplet
    public :: cuda_q4_matmul_device_one
    public :: cuda_q4_matvec_device_group_remote_output
    public :: cuda_q4_matvec_device_remote_input
    public :: cuda_q4_embedding_device
    public :: cuda_q4_embedding_device_batch
    public :: cuda_memory_info

    interface
        function c_context_create(device, context) bind(C, name='fortai_cuda_q8_context_create') &
                result(code)
            import c_int, c_ptr
            integer(c_int), value :: device
            type(c_ptr) :: context
            integer(c_int) :: code
        end function c_context_create

        function c_context_destroy(context) bind(C, name='fortai_cuda_q8_context_destroy') &
                result(code)
            import c_int, c_ptr
            type(c_ptr), value :: context
            integer(c_int) :: code
        end function c_context_destroy

        function c_memory_info(device, free_bytes, total_bytes) &
                bind(C, name='fortai_cuda_memory_info') result(code)
            import c_int, c_size_t
            integer(c_int), value :: device
            integer(c_size_t) :: free_bytes, total_bytes
            integer(c_int) :: code
        end function c_memory_info

        function c_context_set_position(context, position) &
                bind(C, name='fortai_cuda_q8_context_set_position') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: context
            integer(c_int), value :: position
            integer(c_int) :: code
        end function c_context_set_position

        function c_context_synchronize(context) bind(C, name='fortai_cuda_q8_context_synchronize') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: context
            integer(c_int) :: code
        end function c_context_synchronize

        function c_context_stream(context) bind(C, name='fortai_cuda_q8_context_stream') result(stream)
            import c_ptr
            type(c_ptr), value :: context
            type(c_ptr) :: stream
        end function c_context_stream

        function c_context_adopt_stream(context, stream) &
                bind(C, name='fortai_cuda_q8_context_adopt_stream') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: context, stream
            integer(c_int) :: code
        end function c_context_adopt_stream

        function c_context_capture_begin(context) &
                bind(C, name='fortai_cuda_q8_context_capture_begin') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: context
            integer(c_int) :: code
        end function c_context_capture_begin

        function c_context_capture_end(context) &
                bind(C, name='fortai_cuda_q8_context_capture_end') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: context
            integer(c_int) :: code
        end function c_context_capture_end

        function c_context_graph_launch(context) &
                bind(C, name='fortai_cuda_q8_context_graph_launch') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: context
            integer(c_int) :: code
        end function c_context_graph_launch

        function c_weights_upload(context, host_weights, weight_bytes, rows, width, weights) &
                bind(C, name='fortai_cuda_q8_weights_upload') result(code)
            import c_int, c_int8_t, c_ptr, c_size_t
            type(c_ptr), value :: context
            integer(c_int8_t), target, intent(in) :: host_weights(*)
            integer(c_size_t), value :: weight_bytes
            integer(c_int), value :: rows, width
            type(c_ptr) :: weights
            integer(c_int) :: code
        end function c_weights_upload

        function c_weights_destroy(weights) bind(C, name='fortai_cuda_q8_weights_destroy') &
                result(code)
            import c_int, c_ptr
            type(c_ptr), value :: weights
            integer(c_int) :: code
        end function c_weights_destroy

        function c_q4_context_create(first_device, second_device, context) &
                bind(C, name='fortai_cuda_q4_context_create') result(code)
            import c_int, c_ptr
            integer(c_int), value :: first_device, second_device
            type(c_ptr) :: context
            integer(c_int) :: code
        end function c_q4_context_create

        function c_q4_context_destroy(context) bind(C, name='fortai_cuda_q4_context_destroy') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: context
            integer(c_int) :: code
        end function c_q4_context_destroy

        function c_q4_context_synchronize(context) bind(C, name='fortai_cuda_q4_context_synchronize') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: context
            integer(c_int) :: code
        end function c_q4_context_synchronize

        function c_q4_context_stream(context, device_slot) &
                bind(C, name='fortai_cuda_q4_context_stream') result(stream)
            import c_int, c_ptr
            type(c_ptr), value :: context
            integer(c_int), value :: device_slot
            type(c_ptr) :: stream
        end function c_q4_context_stream

        function c_q4_context_set_consumer_stream(context, device_slot, stream) &
                bind(C, name='fortai_cuda_q4_context_set_consumer_stream') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: context, stream
            integer(c_int), value :: device_slot
            integer(c_int) :: code
        end function c_q4_context_set_consumer_stream

        function c_q4_weights_upload(context, value_type, host_weights, weight_bytes, rows, width, &
                device, weights) bind(C, name='fortai_cuda_q4_weights_upload') result(code)
            import c_int, c_int8_t, c_ptr, c_size_t
            type(c_ptr), value :: context
            integer(c_int), value :: value_type
            integer(c_int8_t), target, intent(in) :: host_weights(*)
            integer(c_size_t), value :: weight_bytes
            integer(c_int), value :: rows, width, device
            type(c_ptr) :: weights
            integer(c_int) :: code
        end function c_q4_weights_upload

        function c_q4_weights_destroy(weights) bind(C, name='fortai_cuda_q4_weights_destroy') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: weights
            integer(c_int) :: code
        end function c_q4_weights_destroy

        function c_q4_matvec_host(context, weights, activation, activation_bytes, output, output_bytes, &
                elapsed_ms) bind(C, name='fortai_cuda_q4_matvec_host') result(code)
            import c_float, c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, weights
            real(c_float), target, intent(in) :: activation(*)
            integer(c_size_t), value :: activation_bytes
            real(c_float), target, intent(out) :: output(*)
            integer(c_size_t), value :: output_bytes
            real(c_float), intent(out) :: elapsed_ms
            integer(c_int) :: code
        end function c_q4_matvec_host

        function c_q4_matvec_host_pair(context, first_weights, second_weights, activation, activation_bytes, &
                first_output, first_output_bytes, second_output, second_output_bytes, elapsed_ms) &
                bind(C, name='fortai_cuda_q4_matvec_host_pair') result(code)
            import c_float, c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, first_weights, second_weights
            real(c_float), target, intent(in) :: activation(*)
            integer(c_size_t), value :: activation_bytes
            real(c_float), target, intent(out) :: first_output(*), second_output(*)
            integer(c_size_t), value :: first_output_bytes, second_output_bytes
            real(c_float), intent(out) :: elapsed_ms
            integer(c_int) :: code
        end function c_q4_matvec_host_pair

        function c_q4_matvec_host_triplet(context, first_weights, second_weights, third_weights, activation, &
                activation_bytes, first_output, first_output_bytes, second_output, second_output_bytes, &
                third_output, third_output_bytes, elapsed_ms) bind(C, name='fortai_cuda_q4_matvec_host_triplet') &
                result(code)
            import c_float, c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, first_weights, second_weights, third_weights
            real(c_float), target, intent(in) :: activation(*)
            integer(c_size_t), value :: activation_bytes
            real(c_float), target, intent(out) :: first_output(*), second_output(*), third_output(*)
            integer(c_size_t), value :: first_output_bytes, second_output_bytes, third_output_bytes
            real(c_float), intent(out) :: elapsed_ms
            integer(c_int) :: code
        end function c_q4_matvec_host_triplet

        function c_q4_matvec_device(context, weights, device_activation, activation_elements, device_output, &
                output_elements) bind(C, name='fortai_cuda_q4_matvec_device') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, weights, device_activation, device_output
            integer(c_size_t), value :: activation_elements, output_elements
            integer(c_int) :: code
        end function c_q4_matvec_device

        function c_q4_matvec_device_swiglu(context, gate_weights, up_weights, device_activation, &
                activation_elements, device_output, output_elements) &
                bind(C, name='fortai_cuda_q4_matvec_device_swiglu') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, gate_weights, up_weights, device_activation, device_output
            integer(c_size_t), value :: activation_elements, output_elements
            integer(c_int) :: code
        end function c_q4_matvec_device_swiglu

        function c_q4_matvec_device_swiglu_remote_output(context, gate_weights, up_weights, device_activation, &
                activation_elements, device_output, output_elements) &
                bind(C, name='fortai_cuda_q4_matvec_device_swiglu_remote_output') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, gate_weights, up_weights, device_activation, device_output
            integer(c_size_t), value :: activation_elements, output_elements
            integer(c_int) :: code
        end function c_q4_matvec_device_swiglu_remote_output

        function c_q4_matvec_device_swiglu_down(context, gate_weights, up_weights, down_weights, device_activation, &
                activation_elements, device_output, output_elements) &
                bind(C, name='fortai_cuda_q4_matvec_device_swiglu_down') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, gate_weights, up_weights, down_weights, device_activation, device_output
            integer(c_size_t), value :: activation_elements, output_elements
            integer(c_int) :: code
        end function c_q4_matvec_device_swiglu_down

        function c_q4_matvec_device_group(context, weights, device_activation, activation_elements, &
                device_outputs, output_elements, count) bind(C, name='fortai_cuda_q4_matvec_device_group') &
                result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, weights, device_activation, device_outputs, output_elements
            integer(c_size_t), value :: activation_elements
            integer(c_int), value :: count
            integer(c_int) :: code
        end function c_q4_matvec_device_group

        function c_q4_matvec_device_quad(context, first_weights, second_weights, third_weights, fourth_weights, &
                device_activation, activation_elements, first_output, first_output_elements, second_output, &
                second_output_elements, third_output, third_output_elements, fourth_output, fourth_output_elements) &
                bind(C, name='fortai_cuda_q4_matvec_device_quad') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, first_weights, second_weights, third_weights, fourth_weights
            type(c_ptr), value :: device_activation, first_output, second_output, third_output, fourth_output
            integer(c_size_t), value :: activation_elements, first_output_elements, second_output_elements
            integer(c_size_t), value :: third_output_elements, fourth_output_elements
            integer(c_int) :: code
        end function c_q4_matvec_device_quad

        function c_q4_matmul_device_group(context, weights, device_activation, activation_elements, batch, &
                device_outputs, output_elements, count) bind(C, name='fortai_cuda_q4_matmul_device_group') &
                result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, weights, device_activation, device_outputs, output_elements
            integer(c_size_t), value :: activation_elements
            integer(c_int), value :: batch, count
            integer(c_int) :: code
        end function c_q4_matmul_device_group

        function c_q4_matmul_device(context, weights, device_activation, activation_elements, batch, &
                device_output, output_elements) bind(C, name='fortai_cuda_q4_matmul_device') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, weights, device_activation, device_output
            integer(c_size_t), value :: activation_elements, output_elements
            integer(c_int), value :: batch
            integer(c_int) :: code
        end function c_q4_matmul_device

        function c_q4_matvec_device_group_remote_output(context, weights, device_activation, activation_elements, &
                device_outputs, output_elements, count) &
                bind(C, name='fortai_cuda_q4_matvec_device_group_remote_output') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, weights, device_activation, device_outputs, output_elements
            integer(c_size_t), value :: activation_elements
            integer(c_int), value :: count
            integer(c_int) :: code
        end function c_q4_matvec_device_group_remote_output

        function c_q4_matvec_device_remote_input(context, weights, device_activation, activation_elements, &
                device_output, output_elements) bind(C, name='fortai_cuda_q4_matvec_device_remote_input') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, weights, device_activation, device_output
            integer(c_size_t), value :: activation_elements, output_elements
            integer(c_int) :: code
        end function c_q4_matvec_device_remote_input

        function c_q4_embedding_device(context, weights, token_id, device_output, output_elements) &
                bind(C, name='fortai_cuda_q4_embedding_device') result(code)
            import c_int, c_int64_t, c_ptr, c_size_t
            type(c_ptr), value :: context, weights, device_output
            integer(c_int64_t), value :: token_id
            integer(c_size_t), value :: output_elements
            integer(c_int) :: code
        end function c_q4_embedding_device

        function c_q4_embedding_device_batch(context, weights, host_tokens, batch, device_output, output_elements) &
                bind(C, name='fortai_cuda_q4_embedding_device_batch') result(code)
            import c_int, c_int32_t, c_ptr, c_size_t
            type(c_ptr), value :: context, weights, host_tokens, device_output
            integer(c_int), value :: batch
            integer(c_size_t), value :: output_elements
            integer(c_int) :: code
        end function c_q4_embedding_device_batch

        function c_buffer_create(context, bytes, buffer) &
                bind(C, name='fortai_cuda_q8_device_buffer_create') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context
            integer(c_size_t), value :: bytes
            type(c_ptr) :: buffer
            integer(c_int) :: code
        end function c_buffer_create

        function c_buffer_destroy(context, buffer) &
                bind(C, name='fortai_cuda_q8_device_buffer_destroy') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: context, buffer
            integer(c_int) :: code
        end function c_buffer_destroy

        function c_buffer_upload(context, buffer, host_data, bytes) &
                bind(C, name='fortai_cuda_q8_device_buffer_upload') result(code)
            import c_int, c_int8_t, c_ptr, c_size_t
            type(c_ptr), value :: context, buffer
            integer(c_int8_t), target, intent(in) :: host_data(*)
            integer(c_size_t), value :: bytes
            integer(c_int) :: code
        end function c_buffer_upload

        function c_buffer_upload_ptr(context, buffer, host_data, bytes) &
                bind(C, name='fortai_cuda_q8_device_buffer_upload_ptr') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, buffer, host_data
            integer(c_size_t), value :: bytes
            integer(c_int) :: code
        end function c_buffer_upload_ptr

        function c_buffer_download(context, host_data, buffer, bytes) &
                bind(C, name='fortai_cuda_q8_device_buffer_download') result(code)
            import c_int, c_int8_t, c_ptr, c_size_t
            type(c_ptr), value :: context, buffer
            integer(c_int8_t), target, intent(out) :: host_data(*)
            integer(c_size_t), value :: bytes
            integer(c_int) :: code
        end function c_buffer_download

        function c_buffer_download_ptr(context, host_data, buffer, bytes) &
                bind(C, name='fortai_cuda_q8_device_buffer_download') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, host_data, buffer
            integer(c_size_t), value :: bytes
            integer(c_int) :: code
        end function c_buffer_download_ptr

        function c_matvec_resident(context, weights, activation, output, kernel_ms) &
                bind(C, name='fortai_cuda_q8_matvec_resident') result(code)
            import c_float, c_int, c_ptr
            type(c_ptr), value :: context, weights, activation, output
            real(c_float), intent(out) :: kernel_ms
            integer(c_int) :: code
        end function c_matvec_resident

        function c_matvec_device_f32(context, weights, activation, activation_elements, &
                output, output_elements) bind(C, name='fortai_cuda_q8_matvec_device_f32') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, weights, activation, output
            integer(c_size_t), value :: activation_elements, output_elements
            integer(c_int) :: code
        end function c_matvec_device_f32

        function c_matvec_device_f32_pair(context, first_weights, second_weights, activation, &
                activation_elements, first_output, first_output_elements, second_output, &
                second_output_elements) bind(C, name='fortai_cuda_q8_matvec_device_f32_pair') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, first_weights, second_weights, activation
            integer(c_size_t), value :: activation_elements
            type(c_ptr), value :: first_output, second_output
            integer(c_size_t), value :: first_output_elements, second_output_elements
            integer(c_int) :: code
        end function c_matvec_device_f32_pair

        function c_reserve_matvec_scratch(context, activation_elements, output_elements) &
                bind(C, name='fortai_cuda_q8_reserve_matvec_scratch') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context
            integer(c_size_t), value :: activation_elements, output_elements
            integer(c_int) :: code
        end function c_reserve_matvec_scratch

        function c_matmul_device_f32(context, weights, activation, activation_elements, batch, &
                output, output_elements) bind(C, name='fortai_cuda_q8_matmul_device_f32') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, weights, activation, output
            integer(c_size_t), value :: activation_elements, output_elements
            integer(c_int), value :: batch
            integer(c_int) :: code
        end function c_matmul_device_f32

        function c_embedding_device(context, weights, token_id, output, output_elements) &
                bind(C, name='fortai_cuda_qwen35_embedding_device') result(code)
            import c_int, c_int64_t, c_ptr, c_size_t
            type(c_ptr), value :: context, weights, output
            integer(c_int64_t), value :: token_id
            integer(c_size_t), value :: output_elements
            integer(c_int) :: code
        end function c_embedding_device

        function c_embedding_device_batch(context, weights, host_tokens, batch, output, output_elements) &
                bind(C, name='fortai_cuda_qwen35_embedding_device_batch') result(code)
            import c_int, c_int32_t, c_ptr, c_size_t
            type(c_ptr), value :: context, weights, host_tokens, output
            integer(c_int), value :: batch
            integer(c_size_t), value :: output_elements
            integer(c_int) :: code
        end function c_embedding_device_batch

        function c_qwen35_copy_device(context, device_input, device_output, bytes) &
                bind(C, name='fortai_cuda_qwen35_copy_device') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, device_input, device_output
            integer(c_size_t), value :: bytes
            integer(c_int) :: code
        end function c_qwen35_copy_device

        function c_qwen35_add_device(context, device_left, device_right, device_output, elements) &
                bind(C, name='fortai_cuda_qwen35_add_device') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, device_left, device_right, device_output
            integer(c_size_t), value :: elements
            integer(c_int) :: code
        end function c_qwen35_add_device

        function c_qwen35_copy_column_device(context, device_input, stride, column, device_output, elements) &
                bind(C, name='fortai_cuda_qwen35_copy_column_device') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, device_input, device_output
            integer(c_size_t), value :: stride, elements
            integer(c_int), value :: column
            integer(c_int) :: code
        end function c_qwen35_copy_column_device

        function c_qwen35_add_matrix_device(context, device_left, device_right, device_output, elements) &
                bind(C, name='fortai_cuda_qwen35_add_matrix_device') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, device_left, device_right, device_output
            integer(c_size_t), value :: elements
            integer(c_int) :: code
        end function c_qwen35_add_matrix_device

        function c_qwen35_rms_norm_device(context, device_input, device_weights, device_output, &
                elements, epsilon) bind(C, name='fortai_cuda_qwen35_rms_norm_device') result(code)
            import c_float, c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, device_input, device_weights, device_output
            integer(c_size_t), value :: elements
            real(c_float), value :: epsilon
            integer(c_int) :: code
        end function c_qwen35_rms_norm_device

        function c_qwen35_rms_norm_matrix_device(context, device_input, device_weights, device_output, &
                hidden, batch, epsilon) bind(C, name='fortai_cuda_qwen35_rms_norm_matrix_device') result(code)
            import c_float, c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, device_input, device_weights, device_output
            integer(c_size_t), value :: hidden
            integer(c_int), value :: batch
            real(c_float), value :: epsilon
            integer(c_int) :: code
        end function c_qwen35_rms_norm_matrix_device

        function c_qwen35_silu_product_device(context, device_gate, device_up, elements) &
                bind(C, name='fortai_cuda_qwen35_silu_product_device') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, device_gate, device_up
            integer(c_size_t), value :: elements
            integer(c_int) :: code
        end function c_qwen35_silu_product_device

        function c_qwen35_silu_product_matrix_device(context, device_gate, device_up, elements) &
                bind(C, name='fortai_cuda_qwen35_silu_product_matrix_device') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, device_gate, device_up
            integer(c_size_t), value :: elements
            integer(c_int) :: code
        end function c_qwen35_silu_product_matrix_device

        function c_qwen35_argmax_device(context, device_logits, elements, host_index) &
                bind(C, name='fortai_cuda_qwen35_argmax_device') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, device_logits
            integer(c_size_t), value :: elements
            integer(c_int) :: host_index
            integer(c_int) :: code
        end function c_qwen35_argmax_device

        function c_matvec_host(context, weights, activation, activation_bytes, output, &
                output_bytes, elapsed_ms) bind(C, name='fortai_cuda_q8_matvec_host') result(code)
            import c_float, c_int, c_int8_t, c_ptr, c_size_t
            type(c_ptr), value :: context, weights
            integer(c_int8_t), target, intent(in) :: activation(*)
            integer(c_size_t), value :: activation_bytes
            real(c_float), target, intent(out) :: output(*)
            integer(c_size_t), value :: output_bytes
            real(c_float), intent(out) :: elapsed_ms
            integer(c_int) :: code
        end function c_matvec_host

        function c_matvec_host_pair(context, first_weights, second_weights, activation, &
                activation_bytes, first_output, first_output_bytes, second_output, &
                second_output_bytes, elapsed_ms) bind(C, name='fortai_cuda_q8_matvec_host_pair') &
                result(code)
            import c_float, c_int, c_int8_t, c_ptr, c_size_t
            type(c_ptr), value :: context, first_weights, second_weights
            integer(c_int8_t), target, intent(in) :: activation(*)
            integer(c_size_t), value :: activation_bytes
            real(c_float), target, intent(out) :: first_output(*)
            integer(c_size_t), value :: first_output_bytes
            real(c_float), target, intent(out) :: second_output(*)
            integer(c_size_t), value :: second_output_bytes
            real(c_float), intent(out) :: elapsed_ms
            integer(c_int) :: code
        end function c_matvec_host_pair

        function c_matvec_host_triplet(context, first_weights, second_weights, third_weights, &
                activation, activation_bytes, first_output, first_output_bytes, second_output, &
                second_output_bytes, third_output, third_output_bytes, elapsed_ms) &
                bind(C, name='fortai_cuda_q8_matvec_host_triplet') result(code)
            import c_float, c_int, c_int8_t, c_ptr, c_size_t
            type(c_ptr), value :: context, first_weights, second_weights, third_weights
            integer(c_int8_t), target, intent(in) :: activation(*)
            integer(c_size_t), value :: activation_bytes
            real(c_float), target, intent(out) :: first_output(*)
            integer(c_size_t), value :: first_output_bytes
            real(c_float), target, intent(out) :: second_output(*)
            integer(c_size_t), value :: second_output_bytes
            real(c_float), target, intent(out) :: third_output(*)
            integer(c_size_t), value :: third_output_bytes
            real(c_float), intent(out) :: elapsed_ms
            integer(c_int) :: code
        end function c_matvec_host_triplet

        function c_matvec_host_triplet_contiguous(context, first_weights, second_weights, third_weights, &
                activation, activation_bytes, host_output, host_output_bytes, elapsed_ms) &
                bind(C, name='fortai_cuda_q8_matvec_host_triplet_contiguous') result(code)
            import c_float, c_int, c_int8_t, c_ptr, c_size_t
            type(c_ptr), value :: context, first_weights, second_weights, third_weights
            integer(c_int8_t), target, intent(in) :: activation(*)
            integer(c_size_t), value :: activation_bytes
            real(c_float), target, intent(out) :: host_output(*)
            integer(c_size_t), value :: host_output_bytes
            real(c_float), intent(out) :: elapsed_ms
            integer(c_int) :: code
        end function c_matvec_host_triplet_contiguous

        function c_ffn_host(context, gate_weights, up_weights, down_weights, host_activation, &
                activation_bytes, host_output, output_bytes, elapsed_ms) &
                bind(C, name='fortai_cuda_q8_ffn_host') result(code)
            import c_float, c_int, c_int8_t, c_ptr, c_size_t
            type(c_ptr), value :: context, gate_weights, up_weights, down_weights
            integer(c_int8_t), target, intent(in) :: host_activation(*)
            integer(c_size_t), value :: activation_bytes
            real(c_float), target, intent(out) :: host_output(*)
            integer(c_size_t), value :: output_bytes
            real(c_float), intent(out) :: elapsed_ms
            integer(c_int) :: code
        end function c_ffn_host

        function c_ffn_device(context, gate_weights, up_weights, down_weights, device_activation, &
                activation_elements, device_output, output_elements) &
                bind(C, name='fortai_cuda_q8_ffn_device') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: context, gate_weights, up_weights, down_weights
            type(c_ptr), value :: device_activation, device_output
            integer(c_size_t), value :: activation_elements, output_elements
            integer(c_int) :: code
        end function c_ffn_device

        function c_qwen35_recurrent_create(context, qkv_weights, gate_weights, alpha_weights, &
                beta_weights, output_weights, conv_weights, conv_weight_bytes, conv_size, conv_kernel, &
                ssm_a, ssm_a_bytes, ssm_dt, ssm_dt_bytes, ssm_norm, ssm_norm_bytes, state_size, &
                key_heads, value_heads, head_size, inner_size, norm_epsilon, layer) &
                bind(C, name='fortai_cuda_qwen35_recurrent_create') result(code)
            import c_float, c_int, c_int8_t, c_ptr, c_size_t
            type(c_ptr), value :: context, qkv_weights, gate_weights, alpha_weights, beta_weights
            type(c_ptr), value :: output_weights
            integer(c_int8_t), target, intent(in) :: conv_weights(*)
            integer(c_size_t), value :: conv_weight_bytes
            integer(c_int), value :: conv_size, conv_kernel
            integer(c_int8_t), target, intent(in) :: ssm_a(*)
            integer(c_size_t), value :: ssm_a_bytes
            integer(c_int8_t), target, intent(in) :: ssm_dt(*)
            integer(c_size_t), value :: ssm_dt_bytes
            integer(c_int8_t), target, intent(in) :: ssm_norm(*)
            integer(c_size_t), value :: ssm_norm_bytes
            integer(c_int), value :: state_size, key_heads, value_heads, head_size, inner_size
            real(c_float), value :: norm_epsilon
            type(c_ptr) :: layer
            integer(c_int) :: code
        end function c_qwen35_recurrent_create

        function c_qwen35_recurrent_create_state(context, conv_weights, conv_weight_bytes, conv_size, conv_kernel, &
                ssm_a, ssm_a_bytes, ssm_dt, ssm_dt_bytes, ssm_norm, ssm_norm_bytes, state_size, key_heads, &
                value_heads, head_size, inner_size, norm_epsilon, layer) bind(C, &
                name='fortai_cuda_qwen35_recurrent_create_state') result(code)
            import c_float, c_int, c_int8_t, c_ptr, c_size_t
            type(c_ptr), value :: context
            integer(c_int8_t), target, intent(in) :: conv_weights(*), ssm_a(*), ssm_dt(*), ssm_norm(*)
            integer(c_size_t), value :: conv_weight_bytes, ssm_a_bytes, ssm_dt_bytes, ssm_norm_bytes
            integer(c_int), value :: conv_size, conv_kernel, state_size, key_heads, value_heads, head_size, inner_size
            real(c_float), value :: norm_epsilon
            type(c_ptr) :: layer
            integer(c_int) :: code
        end function c_qwen35_recurrent_create_state

        function c_qwen35_recurrent_destroy(layer) &
                bind(C, name='fortai_cuda_qwen35_recurrent_destroy') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: layer
            integer(c_int) :: code
        end function c_qwen35_recurrent_destroy

        function c_qwen35_recurrent_reset(layer) &
                bind(C, name='fortai_cuda_qwen35_recurrent_reset') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: layer
            integer(c_int) :: code
        end function c_qwen35_recurrent_reset

        function c_qwen35_recurrent_run(layer, host_activation, activation_bytes, host_output, &
                output_bytes, elapsed_ms) bind(C, name='fortai_cuda_qwen35_recurrent_run') result(code)
            import c_float, c_int, c_int8_t, c_ptr, c_size_t
            type(c_ptr), value :: layer
            integer(c_int8_t), target, intent(in) :: host_activation(*)
            integer(c_size_t), value :: activation_bytes
            real(c_float), target, intent(out) :: host_output(*)
            integer(c_size_t), value :: output_bytes
            real(c_float), intent(out) :: elapsed_ms
            integer(c_int) :: code
        end function c_qwen35_recurrent_run

        function c_qwen35_recurrent_run_device(layer, device_activation, activation_elements, &
                device_output, output_elements) bind(C, name='fortai_cuda_qwen35_recurrent_run_device') &
                result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: layer, device_activation, device_output
            integer(c_size_t), value :: activation_elements, output_elements
            integer(c_int) :: code
        end function c_qwen35_recurrent_run_device

        function c_qwen35_recurrent_run_core_device(layer, device_qkv, qkv_elements, device_gate, gate_elements, &
                device_alpha, alpha_elements, device_beta, beta_elements, device_output, output_elements) &
                bind(C, name='fortai_cuda_qwen35_recurrent_run_core_device') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: layer, device_qkv, device_gate, device_alpha, device_beta, device_output
            integer(c_size_t), value :: qkv_elements, gate_elements, alpha_elements, beta_elements, output_elements
            integer(c_int) :: code
        end function c_qwen35_recurrent_run_core_device

        function c_qwen35_recurrent_run_core_device_batch(layer, device_qkv, qkv_elements, device_gate, gate_elements, &
                device_alpha, alpha_elements, device_beta, beta_elements, batch, device_output, output_elements) &
                bind(C, name='fortai_cuda_qwen35_recurrent_run_core_device_batch') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: layer, device_qkv, device_gate, device_alpha, device_beta, device_output
            integer(c_size_t), value :: qkv_elements, gate_elements, alpha_elements, beta_elements, output_elements
            integer(c_int), value :: batch
            integer(c_int) :: code
        end function c_qwen35_recurrent_run_core_device_batch

        function c_qwen35_attention_create(context, query_weights, key_weights, value_weights, &
                output_weights, query_norm, query_norm_bytes, key_norm, key_norm_bytes, heads, &
                key_value_heads, head_size, value_size, max_context, rope_dimension, rope_base, &
                norm_epsilon, cache_key_q8, cache_value_q8, layer) &
                bind(C, name='fortai_cuda_qwen35_attention_create') result(code)
            import c_float, c_int, c_int8_t, c_ptr, c_size_t
            type(c_ptr), value :: context, query_weights, key_weights, value_weights, output_weights
            integer(c_int8_t), target, intent(in) :: query_norm(*), key_norm(*)
            integer(c_size_t), value :: query_norm_bytes, key_norm_bytes
            integer(c_int), value :: heads, key_value_heads, head_size, value_size, max_context
            integer(c_int), value :: rope_dimension
            real(c_float), value :: rope_base, norm_epsilon
            integer(c_int), value :: cache_key_q8, cache_value_q8
            type(c_ptr) :: layer
            integer(c_int) :: code
        end function c_qwen35_attention_create

        function c_qwen35_attention_create_state(context, query_norm, query_norm_bytes, key_norm, key_norm_bytes, &
                heads, key_value_heads, head_size, value_size, max_context, rope_dimension, rope_base, &
                norm_epsilon, cache_key_q8, cache_value_q8, layer) &
                bind(C, name='fortai_cuda_qwen35_attention_create_state') result(code)
            import c_float, c_int, c_int8_t, c_ptr, c_size_t
            type(c_ptr), value :: context
            integer(c_int8_t), target, intent(in) :: query_norm(*), key_norm(*)
            integer(c_size_t), value :: query_norm_bytes, key_norm_bytes
            integer(c_int), value :: heads, key_value_heads, head_size, value_size, max_context, rope_dimension
            real(c_float), value :: rope_base, norm_epsilon
            integer(c_int), value :: cache_key_q8, cache_value_q8
            type(c_ptr) :: layer
            integer(c_int) :: code
        end function c_qwen35_attention_create_state

        function c_qwen35_attention_destroy(layer) &
                bind(C, name='fortai_cuda_qwen35_attention_destroy') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: layer
            integer(c_int) :: code
        end function c_qwen35_attention_destroy

        function c_qwen35_attention_reset(layer) &
                bind(C, name='fortai_cuda_qwen35_attention_reset') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: layer
            integer(c_int) :: code
        end function c_qwen35_attention_reset

        function c_qwen35_attention_run_device(layer, device_activation, activation_elements, position, &
                device_output, output_elements) bind(C, name='fortai_cuda_qwen35_attention_run_device') &
                result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: layer, device_activation, device_output
            integer(c_size_t), value :: activation_elements, output_elements
            integer(c_int), value :: position
            integer(c_int) :: code
        end function c_qwen35_attention_run_device

        function c_qwen35_attention_run_core_device(layer, device_query, query_elements, device_key, key_elements, &
                device_value, value_elements, position, device_output, output_elements) bind(C, &
                name='fortai_cuda_qwen35_attention_run_core_device') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: layer, device_query, device_key, device_value, device_output
            integer(c_size_t), value :: query_elements, key_elements, value_elements, output_elements
            integer(c_int), value :: position
            integer(c_int) :: code
        end function c_qwen35_attention_run_core_device

        function c_qwen35_attention_run_core_device_batch(layer, device_query, query_elements, device_key, key_elements, &
                device_value, value_elements, position, batch, device_output, output_elements) bind(C, &
                name='fortai_cuda_qwen35_attention_run_core_device_batch') result(code)
            import c_int, c_ptr, c_size_t
            type(c_ptr), value :: layer, device_query, device_key, device_value, device_output
            integer(c_size_t), value :: query_elements, key_elements, value_elements, output_elements
            integer(c_int), value :: position, batch
            integer(c_int) :: code
        end function c_qwen35_attention_run_core_device_batch

        function c_last_error(context) bind(C, name='fortai_cuda_q8_last_error') result(message)
            import c_ptr
            type(c_ptr), value :: context
            type(c_ptr) :: message
        end function c_last_error
    end interface

contains

    subroutine cuda_memory_info(device, free_bytes, total_bytes, stat)
        integer, intent(in) :: device
        integer(c_size_t), intent(out) :: free_bytes, total_bytes
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        free_bytes = 0_c_size_t
        total_bytes = 0_c_size_t
        code = c_memory_info(int(device, c_int), free_bytes, total_bytes)
        call stat%clear()
        if (code /= FORTAI_CUDA_OK) then
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA memory query unavailable')
        end if
    end subroutine cuda_memory_info

    subroutine cuda_q4_context_create(self, first_device, second_device, stat)
        class(cuda_q4_context_t), intent(inout) :: self
        integer, intent(in) :: first_device, second_device
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (c_associated(self%handle)) call self%destroy(stat)
        code = c_q4_context_create(int(first_device, c_int), int(second_device, c_int), self%handle)
        if (code /= FORTAI_CUDA_OK) then
            self%handle = c_null_ptr
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA Q4 context creation failed')
        end if
    end subroutine cuda_q4_context_create

    subroutine cuda_q4_context_destroy(self, stat)
        class(cuda_q4_context_t), intent(inout) :: self
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle)) return
        code = c_q4_context_destroy(self%handle)
        self%handle = c_null_ptr
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q4 context destruction failed')
    end subroutine cuda_q4_context_destroy

    subroutine cuda_q4_context_synchronize(self, stat)
        class(cuda_q4_context_t), intent(in) :: self
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 context synchronization arguments')
            return
        end if
        code = c_q4_context_synchronize(self%handle)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q4 context synchronization failed')
    end subroutine cuda_q4_context_synchronize

    function cuda_q4_context_stream(self, device_slot) result(stream)
        class(cuda_q4_context_t), intent(in) :: self
        integer, intent(in) :: device_slot
        type(c_ptr) :: stream

        if (c_associated(self%handle) .and. device_slot >= 0 .and. device_slot <= 1) then
            stream = c_q4_context_stream(self%handle, int(device_slot, c_int))
        else
            stream = c_null_ptr
        end if
    end function cuda_q4_context_stream

    subroutine cuda_q4_context_set_consumer_stream(self, device_slot, stream, stat)
        class(cuda_q4_context_t), intent(in) :: self
        integer, intent(in) :: device_slot
        type(c_ptr), intent(in) :: stream
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle) .or. device_slot < 0 .or. device_slot > 1) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 consumer stream arguments')
            return
        end if
        code = c_q4_context_set_consumer_stream(self%handle, int(device_slot, c_int), stream)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q4 consumer stream setup failed')
    end subroutine cuda_q4_context_set_consumer_stream

    subroutine cuda_q4_weights_upload(self, context, value_type, host_weights, weight_bytes, rows, width, &
            device, stat)
        class(cuda_q4_weights_t), intent(inout) :: self
        class(cuda_q4_context_t), intent(in) :: context
        integer, intent(in) :: value_type, rows, width, device
        integer(c_int8_t), contiguous, target, intent(in) :: host_weights(:)
        integer(c_size_t), intent(in) :: weight_bytes
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. size(host_weights) <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 weight upload arguments')
            return
        end if
        if (c_associated(self%handle)) call self%destroy(stat)
        code = c_q4_weights_upload(context%handle, int(value_type, c_int), host_weights, weight_bytes, &
            int(rows, c_int), int(width, c_int), int(device, c_int), self%handle)
        if (code /= FORTAI_CUDA_OK) then
            self%handle = c_null_ptr
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA Q4 weight upload failed')
        end if
    end subroutine cuda_q4_weights_upload

    subroutine cuda_q4_weights_destroy(self, stat)
        class(cuda_q4_weights_t), intent(inout) :: self
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle)) return
        code = c_q4_weights_destroy(self%handle)
        self%handle = c_null_ptr
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q4 weight destruction failed')
    end subroutine cuda_q4_weights_destroy

    subroutine cuda_q4_matvec_host(context, weights, activation, output, elapsed_ms, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: weights
        real(c_float), contiguous, target, intent(in) :: activation(:)
        real(c_float), contiguous, target, intent(out) :: output(:)
        real(c_float), intent(out) :: elapsed_ms
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        elapsed_ms = 0.0_c_float
        if (.not. c_associated(context%handle) .or. .not. c_associated(weights%handle)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 matvec arguments')
            return
        end if
        code = c_q4_matvec_host(context%handle, weights%handle, activation, &
            int(size(activation) * storage_size(activation(1)) / 8, c_size_t), output, &
            int(size(output) * storage_size(output(1)) / 8, c_size_t), elapsed_ms)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, 'CUDA Q4 matvec failed')
    end subroutine cuda_q4_matvec_host

    subroutine cuda_q4_matvec_host_pair(context, first_weights, second_weights, activation, first_output, &
            second_output, elapsed_ms, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: first_weights, second_weights
        real(c_float), contiguous, target, intent(in) :: activation(:)
        real(c_float), contiguous, target, intent(out) :: first_output(:), second_output(:)
        real(c_float), intent(out) :: elapsed_ms
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        elapsed_ms = 0.0_c_float
        if (.not. c_associated(context%handle) .or. .not. c_associated(first_weights%handle) .or. &
            .not. c_associated(second_weights%handle)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 paired matvec arguments')
            return
        end if
        code = c_q4_matvec_host_pair(context%handle, first_weights%handle, second_weights%handle, activation, &
            int(size(activation) * storage_size(activation(1)) / 8, c_size_t), first_output, &
            int(size(first_output) * storage_size(first_output(1)) / 8, c_size_t), second_output, &
            int(size(second_output) * storage_size(second_output(1)) / 8, c_size_t), elapsed_ms)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, 'CUDA Q4 paired matvec failed')
    end subroutine cuda_q4_matvec_host_pair

    subroutine cuda_q4_matvec_host_triplet(context, first_weights, second_weights, third_weights, activation, &
            first_output, second_output, third_output, elapsed_ms, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: first_weights, second_weights, third_weights
        real(c_float), contiguous, target, intent(in) :: activation(:)
        real(c_float), contiguous, target, intent(out) :: first_output(:), second_output(:), third_output(:)
        real(c_float), intent(out) :: elapsed_ms
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        elapsed_ms = 0.0_c_float
        if (.not. c_associated(context%handle) .or. .not. c_associated(first_weights%handle) .or. &
            .not. c_associated(second_weights%handle) .or. .not. c_associated(third_weights%handle)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 triplet matvec arguments')
            return
        end if
        code = c_q4_matvec_host_triplet(context%handle, first_weights%handle, second_weights%handle, &
            third_weights%handle, activation, int(size(activation) * storage_size(activation(1)) / 8, c_size_t), &
            first_output, int(size(first_output) * storage_size(first_output(1)) / 8, c_size_t), second_output, &
            int(size(second_output) * storage_size(second_output(1)) / 8, c_size_t), third_output, &
            int(size(third_output) * storage_size(third_output(1)) / 8, c_size_t), elapsed_ms)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, 'CUDA Q4 triplet matvec failed')
    end subroutine cuda_q4_matvec_host_triplet

    subroutine cuda_q4_matvec_device(context, weights, device_activation, activation_elements, device_output, &
            output_elements, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: weights
        type(c_ptr), intent(in) :: device_activation, device_output
        integer(c_size_t), intent(in) :: activation_elements, output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(weights%handle) .or. &
            .not. c_associated(device_activation) .or. .not. c_associated(device_output)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 resident matvec arguments')
            return
        end if
        code = c_q4_matvec_device(context%handle, weights%handle, device_activation, activation_elements, &
            device_output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q4 resident matvec failed')
    end subroutine cuda_q4_matvec_device

    subroutine cuda_q4_matvec_device_swiglu(context, gate_weights, up_weights, device_activation, &
            activation_elements, device_output, output_elements, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: gate_weights, up_weights
        type(c_ptr), intent(in) :: device_activation, device_output
        integer(c_size_t), intent(in) :: activation_elements, output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(gate_weights%handle) .or. &
            .not. c_associated(up_weights%handle) .or. .not. c_associated(device_activation) .or. &
            .not. c_associated(device_output)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 fused SwiGLU arguments')
            return
        end if
        code = c_q4_matvec_device_swiglu(context%handle, gate_weights%handle, up_weights%handle, &
            device_activation, activation_elements, device_output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q4 fused SwiGLU matvec failed')
    end subroutine cuda_q4_matvec_device_swiglu

    subroutine cuda_q4_matvec_device_swiglu_remote_output(context, gate_weights, up_weights, device_activation, &
            activation_elements, device_output, output_elements, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: gate_weights, up_weights
        type(c_ptr), intent(in) :: device_activation, device_output
        integer(c_size_t), intent(in) :: activation_elements, output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(gate_weights%handle) .or. &
            .not. c_associated(up_weights%handle) .or. .not. c_associated(device_activation) .or. &
            .not. c_associated(device_output)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 fused remote SwiGLU arguments')
            return
        end if
        code = c_q4_matvec_device_swiglu_remote_output(context%handle, gate_weights%handle, &
            up_weights%handle, device_activation, activation_elements, device_output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q4 fused remote SwiGLU matvec failed')
    end subroutine cuda_q4_matvec_device_swiglu_remote_output

    subroutine cuda_q4_matvec_device_swiglu_down(context, gate_weights, up_weights, down_weights, device_activation, &
            activation_elements, device_output, output_elements, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: gate_weights, up_weights, down_weights
        type(c_ptr), intent(in) :: device_activation, device_output
        integer(c_size_t), intent(in) :: activation_elements, output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(gate_weights%handle) .or. &
            .not. c_associated(up_weights%handle) .or. .not. c_associated(down_weights%handle) .or. &
            .not. c_associated(device_activation) .or. .not. c_associated(device_output)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 fused SwiGLU/down arguments')
            return
        end if
        code = c_q4_matvec_device_swiglu_down(context%handle, gate_weights%handle, up_weights%handle, &
            down_weights%handle, device_activation, activation_elements, device_output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q4 fused SwiGLU/down matvec failed')
    end subroutine cuda_q4_matvec_device_swiglu_down

    subroutine cuda_q4_matvec_device_pair(context, first_weights, second_weights, device_activation, &
            activation_elements, first_output, first_output_elements, second_output, second_output_elements, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: first_weights, second_weights
        type(c_ptr), intent(in) :: device_activation, first_output, second_output
        integer(c_size_t), intent(in) :: activation_elements, first_output_elements, second_output_elements
        type(status_t), intent(out) :: stat
        type(c_ptr), target :: handles(2), outputs(2)
        integer(c_size_t), target :: sizes(2)
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(first_weights%handle) .or. &
            .not. c_associated(second_weights%handle) .or. .not. c_associated(device_activation) .or. &
            .not. c_associated(first_output) .or. .not. c_associated(second_output)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 resident pair arguments')
            return
        end if
        handles(1) = first_weights%handle
        handles(2) = second_weights%handle
        outputs(1) = first_output
        outputs(2) = second_output
        sizes = [first_output_elements, second_output_elements]
        code = c_q4_matvec_device_group(context%handle, c_loc(handles), device_activation, activation_elements, &
            c_loc(outputs), c_loc(sizes), 2_c_int)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q4 resident paired matvec failed')
    end subroutine cuda_q4_matvec_device_pair

    subroutine cuda_q4_matvec_device_triplet(context, first_weights, second_weights, third_weights, &
            device_activation, activation_elements, first_output, first_output_elements, second_output, &
            second_output_elements, third_output, third_output_elements, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: first_weights, second_weights, third_weights
        type(c_ptr), intent(in) :: device_activation, first_output, second_output, third_output
        integer(c_size_t), intent(in) :: activation_elements, first_output_elements, second_output_elements, &
            third_output_elements
        type(status_t), intent(out) :: stat
        type(c_ptr), target :: handles(3), outputs(3)
        integer(c_size_t), target :: sizes(3)
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(first_weights%handle) .or. &
            .not. c_associated(second_weights%handle) .or. .not. c_associated(third_weights%handle) .or. &
            .not. c_associated(device_activation) .or. .not. c_associated(first_output) .or. &
            .not. c_associated(second_output) .or. .not. c_associated(third_output)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 resident triplet arguments')
            return
        end if
        handles(1) = first_weights%handle
        handles(2) = second_weights%handle
        handles(3) = third_weights%handle
        outputs(1) = first_output
        outputs(2) = second_output
        outputs(3) = third_output
        sizes = [first_output_elements, second_output_elements, third_output_elements]
        code = c_q4_matvec_device_group(context%handle, c_loc(handles), device_activation, activation_elements, &
            c_loc(outputs), c_loc(sizes), 3_c_int)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q4 resident triplet matvec failed')
    end subroutine cuda_q4_matvec_device_triplet

    subroutine cuda_q4_matvec_device_quad(context, first_weights, second_weights, third_weights, fourth_weights, &
            device_activation, activation_elements, first_output, first_output_elements, second_output, &
            second_output_elements, third_output, third_output_elements, fourth_output, fourth_output_elements, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: first_weights, second_weights, third_weights, fourth_weights
        type(c_ptr), intent(in) :: device_activation, first_output, second_output, third_output, fourth_output
        integer(c_size_t), intent(in) :: activation_elements, first_output_elements, second_output_elements, &
            third_output_elements, fourth_output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(first_weights%handle) .or. &
            .not. c_associated(second_weights%handle) .or. .not. c_associated(third_weights%handle) .or. &
            .not. c_associated(fourth_weights%handle) .or. .not. c_associated(device_activation) .or. &
            .not. c_associated(first_output) .or. .not. c_associated(second_output) .or. &
            .not. c_associated(third_output) .or. .not. c_associated(fourth_output)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 resident quad arguments')
            return
        end if
        code = c_q4_matvec_device_quad(context%handle, first_weights%handle, second_weights%handle, &
            third_weights%handle, fourth_weights%handle, device_activation, activation_elements, first_output, &
            first_output_elements, second_output, second_output_elements, third_output, third_output_elements, &
            fourth_output, fourth_output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q4 resident quad matvec failed')
    end subroutine cuda_q4_matvec_device_quad

    subroutine cuda_q4_matmul_device_pair(context, first_weights, second_weights, device_activation, &
            activation_elements, batch, first_output, first_output_elements, second_output, &
            second_output_elements, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: first_weights, second_weights
        type(c_ptr), intent(in) :: device_activation, first_output, second_output
        integer(c_size_t), intent(in) :: activation_elements, first_output_elements, second_output_elements
        integer, intent(in) :: batch
        type(status_t), intent(out) :: stat
        type(c_ptr), target :: handles(2), outputs(2)
        integer(c_size_t), target :: sizes(2)
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(first_weights%handle) .or. &
            .not. c_associated(second_weights%handle) .or. .not. c_associated(device_activation) .or. &
            .not. c_associated(first_output) .or. .not. c_associated(second_output) .or. batch <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 batched pair arguments')
            return
        end if
        handles = [first_weights%handle, second_weights%handle]
        outputs = [first_output, second_output]
        sizes = [first_output_elements, second_output_elements]
        code = c_q4_matmul_device_group(context%handle, c_loc(handles), device_activation, activation_elements, &
            int(batch, c_int), c_loc(outputs), c_loc(sizes), 2_c_int)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, 'CUDA Q4 batched pair failed')
    end subroutine cuda_q4_matmul_device_pair

    subroutine cuda_q4_matmul_device_one(context, weights, device_activation, activation_elements, batch, &
            device_output, output_elements, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: weights
        type(c_ptr), intent(in) :: device_activation, device_output
        integer(c_size_t), intent(in) :: activation_elements, output_elements
        integer, intent(in) :: batch
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(weights%handle) .or. &
            .not. c_associated(device_activation) .or. .not. c_associated(device_output) .or. batch <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 batched matrix arguments')
            return
        end if
        code = c_q4_matmul_device(context%handle, weights%handle, device_activation, activation_elements, &
            int(batch, c_int), device_output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, 'CUDA Q4 batched matrix failed')
    end subroutine cuda_q4_matmul_device_one

    subroutine cuda_q4_matmul_device_triplet(context, first_weights, second_weights, third_weights, &
            device_activation, activation_elements, batch, first_output, first_output_elements, second_output, &
            second_output_elements, third_output, third_output_elements, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: first_weights, second_weights, third_weights
        type(c_ptr), intent(in) :: device_activation, first_output, second_output, third_output
        integer(c_size_t), intent(in) :: activation_elements, first_output_elements, second_output_elements, &
            third_output_elements
        integer, intent(in) :: batch
        type(status_t), intent(out) :: stat
        type(c_ptr), target :: handles(3), outputs(3)
        integer(c_size_t), target :: sizes(3)
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(first_weights%handle) .or. &
            .not. c_associated(second_weights%handle) .or. .not. c_associated(third_weights%handle) .or. &
            .not. c_associated(device_activation) .or. .not. c_associated(first_output) .or. &
            .not. c_associated(second_output) .or. .not. c_associated(third_output) .or. batch <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 batched triplet arguments')
            return
        end if
        handles = [first_weights%handle, second_weights%handle, third_weights%handle]
        outputs = [first_output, second_output, third_output]
        sizes = [first_output_elements, second_output_elements, third_output_elements]
        code = c_q4_matmul_device_group(context%handle, c_loc(handles), device_activation, activation_elements, &
            int(batch, c_int), c_loc(outputs), c_loc(sizes), 3_c_int)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, 'CUDA Q4 batched triplet failed')
    end subroutine cuda_q4_matmul_device_triplet

    subroutine cuda_q4_matvec_device_group_remote_output(context, first_weights, second_weights, third_weights, &
            count, device_activation, activation_elements, first_output, first_output_elements, second_output, &
            second_output_elements, third_output, third_output_elements, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: first_weights, second_weights, third_weights
        integer, intent(in) :: count
        type(c_ptr), intent(in) :: device_activation, first_output, second_output, third_output
        integer(c_size_t), intent(in) :: activation_elements, first_output_elements, second_output_elements, &
            third_output_elements
        type(status_t), intent(out) :: stat
        type(c_ptr), target :: handles(3), outputs(3)
        integer(c_size_t), target :: sizes(3)
        integer(c_int) :: code

        call stat%clear()
        if (count < 1 .or. count > 3) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 remote-output group size')
            return
        end if
        if (.not. c_associated(context%handle) .or. .not. c_associated(first_weights%handle) .or. &
            .not. c_associated(device_activation) .or. .not. c_associated(first_output)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 remote-output arguments')
            return
        end if
        handles = [first_weights%handle, second_weights%handle, third_weights%handle]
        outputs = [first_output, second_output, third_output]
        sizes = [first_output_elements, second_output_elements, third_output_elements]
        if (count >= 2) then
            if (.not. c_associated(second_weights%handle) .or. .not. c_associated(second_output)) then
                call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 remote-output pair arguments')
                return
            end if
        end if
        if (count >= 3) then
            if (.not. c_associated(third_weights%handle) .or. .not. c_associated(third_output)) then
                call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 remote-output triplet arguments')
                return
            end if
        end if
        code = c_q4_matvec_device_group_remote_output(context%handle, c_loc(handles), device_activation, &
            activation_elements, c_loc(outputs), c_loc(sizes), int(count, c_int))
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q4 resident remote-output group failed')
    end subroutine cuda_q4_matvec_device_group_remote_output

    subroutine cuda_q4_matvec_device_remote_input(context, weights, device_activation, activation_elements, &
            device_output, output_elements, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: weights
        type(c_ptr), intent(in) :: device_activation, device_output
        integer(c_size_t), intent(in) :: activation_elements, output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(weights%handle) .or. &
            .not. c_associated(device_activation) .or. .not. c_associated(device_output)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 remote-input arguments')
            return
        end if
        code = c_q4_matvec_device_remote_input(context%handle, weights%handle, device_activation, &
            activation_elements, device_output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q4 resident remote-input matvec failed')
    end subroutine cuda_q4_matvec_device_remote_input

    subroutine cuda_q4_embedding_device(context, weights, token_id, device_output, output_elements, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: weights
        integer(c_int64_t), intent(in) :: token_id
        type(c_ptr), intent(in) :: device_output
        integer(c_size_t), intent(in) :: output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(weights%handle) .or. &
            .not. c_associated(device_output)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 resident embedding arguments')
            return
        end if
        code = c_q4_embedding_device(context%handle, weights%handle, token_id, device_output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q4 resident embedding failed')
    end subroutine cuda_q4_embedding_device

    subroutine cuda_q4_embedding_device_batch(context, weights, host_tokens, batch, device_output, output_elements, stat)
        class(cuda_q4_context_t), intent(in) :: context
        class(cuda_q4_weights_t), intent(in) :: weights
        integer(c_int32_t), contiguous, target, intent(in) :: host_tokens(:)
        integer, intent(in) :: batch
        type(c_ptr), intent(in) :: device_output
        integer(c_size_t), intent(in) :: output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(weights%handle) .or. &
            .not. c_associated(device_output) .or. batch <= 0 .or. size(host_tokens) < batch) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q4 batched embedding arguments')
            return
        end if
        code = c_q4_embedding_device_batch(context%handle, weights%handle, c_loc(host_tokens), int(batch, c_int), &
            device_output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, 'CUDA Q4 batched embedding failed')
    end subroutine cuda_q4_embedding_device_batch

    subroutine cuda_q8_context_create(self, device, stat)
        class(cuda_q8_context_t), intent(inout) :: self
        integer, intent(in) :: device
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (c_associated(self%handle)) call self%destroy(stat)
        code = c_context_create(int(device, c_int), self%handle)
        if (code /= FORTAI_CUDA_OK) then
            self%handle = c_null_ptr
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA Q8 context creation failed')
        end if
    end subroutine cuda_q8_context_create

    subroutine cuda_q8_context_destroy(self, stat)
        class(cuda_q8_context_t), intent(inout) :: self
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle)) return
        code = c_context_destroy(self%handle)
        self%handle = c_null_ptr
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q8 context destruction failed')
    end subroutine cuda_q8_context_destroy

    subroutine cuda_q8_context_set_position(self, position, stat)
        class(cuda_q8_context_t), intent(in) :: self
        integer, intent(in) :: position
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle) .or. position < 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA device position')
            return
        end if
        code = c_context_set_position(self%handle, int(position, c_int))
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA device position update failed')
    end subroutine cuda_q8_context_set_position

    subroutine cuda_q8_context_synchronize(self, stat)
        class(cuda_q8_context_t), intent(in) :: self
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA context synchronization arguments')
            return
        end if
        code = c_context_synchronize(self%handle)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA context synchronization failed')
    end subroutine cuda_q8_context_synchronize

    subroutine cuda_q8_context_adopt_stream(self, stream, stat)
        class(cuda_q8_context_t), intent(in) :: self
        type(c_ptr), intent(in) :: stream
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle) .or. .not. c_associated(stream)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA stream hand-off arguments')
            return
        end if
        code = c_context_adopt_stream(self%handle, stream)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA stream hand-off failed')
    end subroutine cuda_q8_context_adopt_stream

    function cuda_q8_context_stream(self) result(stream)
        class(cuda_q8_context_t), intent(in) :: self
        type(c_ptr) :: stream

        if (c_associated(self%handle)) then
            stream = c_context_stream(self%handle)
        else
            stream = c_null_ptr
        end if
    end function cuda_q8_context_stream

    subroutine cuda_q8_context_capture_begin(self, stat)
        class(cuda_q8_context_t), intent(in) :: self
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA graph context')
            return
        end if
        code = c_context_capture_begin(self%handle)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA graph capture begin failed')
    end subroutine cuda_q8_context_capture_begin

    subroutine cuda_q8_context_capture_end(self, stat)
        class(cuda_q8_context_t), intent(in) :: self
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA graph context')
            return
        end if
        code = c_context_capture_end(self%handle)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA graph capture end failed')
    end subroutine cuda_q8_context_capture_end

    subroutine cuda_q8_context_graph_launch(self, stat)
        class(cuda_q8_context_t), intent(in) :: self
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA graph context')
            return
        end if
        code = c_context_graph_launch(self%handle)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA graph launch failed')
    end subroutine cuda_q8_context_graph_launch

    subroutine cuda_q8_weights_upload(self, context, host_weights, weight_bytes, rows, width, stat)
        class(cuda_q8_weights_t), intent(inout) :: self
        class(cuda_q8_context_t), intent(in) :: context
        integer(c_int8_t), contiguous, target, intent(in) :: host_weights(:)
        integer(c_size_t), intent(in) :: weight_bytes
        integer, intent(in) :: rows, width
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. size(host_weights) <= 0 .or. &
            rows <= 0 .or. width <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q8 weight upload')
            return
        end if
        if (c_associated(self%handle)) call self%destroy(stat)
        code = c_weights_upload(context%handle, host_weights, weight_bytes, int(rows, c_int), &
            int(width, c_int), self%handle)
        if (code /= FORTAI_CUDA_OK) then
            self%handle = c_null_ptr
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA Q8 weight upload failed')
        end if
    end subroutine cuda_q8_weights_upload

    subroutine cuda_q8_weights_destroy(self, stat)
        class(cuda_q8_weights_t), intent(inout) :: self
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle)) return
        code = c_weights_destroy(self%handle)
        self%handle = c_null_ptr
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q8 weight destruction failed')
    end subroutine cuda_q8_weights_destroy

    subroutine cuda_q8_allocate_buffer(self, bytes, buffer, stat)
        class(cuda_q8_context_t), intent(in) :: self
        integer(c_size_t), intent(in) :: bytes
        type(c_ptr), intent(out) :: buffer
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        buffer = c_null_ptr
        if (.not. c_associated(self%handle) .or. bytes <= 0_c_size_t) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q8 device buffer')
            return
        end if
        code = c_buffer_create(self%handle, bytes, buffer)
        if (code /= FORTAI_CUDA_OK) then
            buffer = c_null_ptr
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA Q8 device buffer allocation failed')
        end if
    end subroutine cuda_q8_allocate_buffer

    subroutine cuda_q8_free_buffer(self, buffer, stat)
        class(cuda_q8_context_t), intent(in) :: self
        type(c_ptr), intent(inout) :: buffer
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle) .or. .not. c_associated(buffer)) return
        code = c_buffer_destroy(self%handle, buffer)
        buffer = c_null_ptr
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q8 device buffer destruction failed')
    end subroutine cuda_q8_free_buffer

    subroutine cuda_q8_upload(self, buffer, host_data, bytes, stat)
        class(cuda_q8_context_t), intent(in) :: self
        type(c_ptr), intent(in) :: buffer
        integer(c_int8_t), contiguous, target, intent(in) :: host_data(:)
        integer(c_size_t), intent(in) :: bytes
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle) .or. .not. c_associated(buffer) .or. &
            size(host_data) <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q8 upload')
            return
        end if
        code = c_buffer_upload(self%handle, buffer, host_data, bytes)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q8 device upload failed')
    end subroutine cuda_q8_upload

    subroutine cuda_q8_upload_real(self, buffer, host_data, stat)
        class(cuda_q8_context_t), intent(in) :: self
        type(c_ptr), intent(in) :: buffer
        real(c_float), contiguous, target, intent(in) :: host_data(:)
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle) .or. .not. c_associated(buffer) .or. &
            size(host_data) <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA real device upload')
            return
        end if
        code = c_buffer_upload_ptr(self%handle, buffer, c_loc(host_data), &
            int(size(host_data), c_size_t) * int(storage_size(host_data(1)) / 8, c_size_t))
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA real device upload failed')
    end subroutine cuda_q8_upload_real

    subroutine cuda_q8_download(self, buffer, host_data, bytes, stat)
        class(cuda_q8_context_t), intent(in) :: self
        type(c_ptr), intent(in) :: buffer
        integer(c_int8_t), contiguous, target, intent(out) :: host_data(:)
        integer(c_size_t), intent(in) :: bytes
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle) .or. .not. c_associated(buffer) .or. &
            size(host_data) <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q8 download')
            return
        end if
        code = c_buffer_download(self%handle, host_data, buffer, bytes)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q8 device download failed')
    end subroutine cuda_q8_download

    subroutine cuda_q8_download_real(self, buffer, host_data, stat)
        class(cuda_q8_context_t), intent(in) :: self
        type(c_ptr), intent(in) :: buffer
        real(c_float), contiguous, target, intent(out) :: host_data(:)
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle) .or. .not. c_associated(buffer) .or. &
            size(host_data) <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q8 real download')
            return
        end if
        code = c_buffer_download_ptr(self%handle, c_loc(host_data), buffer, &
            int(size(host_data), c_size_t) * int(storage_size(host_data(1)) / 8, c_size_t))
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q8 real download failed')
    end subroutine cuda_q8_download_real

    subroutine cuda_q8_matvec_resident(context, weights, activation, output, kernel_ms, stat)
        class(cuda_q8_context_t), intent(in) :: context
        class(cuda_q8_weights_t), intent(in) :: weights
        type(c_ptr), intent(in) :: activation, output
        real(c_float), intent(out) :: kernel_ms
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        kernel_ms = 0.0_c_float
        if (.not. c_associated(context%handle) .or. .not. c_associated(weights%handle) .or. &
            .not. c_associated(activation) .or. .not. c_associated(output)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q8 resident matvec')
            return
        end if
        code = c_matvec_resident(context%handle, weights%handle, activation, output, kernel_ms)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q8 resident matvec failed')
    end subroutine cuda_q8_matvec_resident

    subroutine cuda_q8_reserve_matvec_scratch(context, activation_elements, output_elements, stat)
        class(cuda_q8_context_t), intent(in) :: context
        integer(c_size_t), intent(in) :: activation_elements, output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. activation_elements <= 0_c_size_t .or. &
            output_elements <= 0_c_size_t) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA F32 matvec scratch reservation')
            return
        end if
        code = c_reserve_matvec_scratch(context%handle, activation_elements, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA F32 matvec scratch reservation failed')
    end subroutine cuda_q8_reserve_matvec_scratch

    subroutine cuda_q8_matvec_device_f32(context, weights, activation, activation_elements, &
            output, output_elements, stat)
        class(cuda_q8_context_t), intent(in) :: context
        class(cuda_q8_weights_t), intent(in) :: weights
        type(c_ptr), intent(in) :: activation, output
        integer(c_size_t), intent(in) :: activation_elements, output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(weights%handle) .or. &
            .not. c_associated(activation) .or. .not. c_associated(output) .or. &
            activation_elements <= 0_c_size_t .or. output_elements <= 0_c_size_t) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA device F32 matvec')
            return
        end if
        code = c_matvec_device_f32(context%handle, weights%handle, activation, activation_elements, &
            output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA device F32 matvec failed')
    end subroutine cuda_q8_matvec_device_f32

    subroutine cuda_q8_matvec_device_f32_pair(context, first_weights, second_weights, activation, &
            activation_elements, first_output, first_output_elements, second_output, &
            second_output_elements, stat)
        class(cuda_q8_context_t), intent(in) :: context
        class(cuda_q8_weights_t), intent(in) :: first_weights, second_weights
        type(c_ptr), intent(in) :: activation, first_output, second_output
        integer(c_size_t), intent(in) :: activation_elements, first_output_elements, second_output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(first_weights%handle) .or. &
            .not. c_associated(second_weights%handle) .or. .not. c_associated(activation) .or. &
            .not. c_associated(first_output) .or. .not. c_associated(second_output) .or. &
            activation_elements <= 0_c_size_t .or. first_output_elements <= 0_c_size_t .or. &
            second_output_elements <= 0_c_size_t) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA device F32 matvec pair')
            return
        end if
        code = c_matvec_device_f32_pair(context%handle, first_weights%handle, second_weights%handle, &
            activation, activation_elements, first_output, first_output_elements, second_output, &
            second_output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA device F32 matvec pair failed')
    end subroutine cuda_q8_matvec_device_f32_pair

    subroutine cuda_q8_matmul_device_f32(context, weights, activation, activation_elements, batch, &
            output, output_elements, stat)
        class(cuda_q8_context_t), intent(in) :: context
        class(cuda_q8_weights_t), intent(in) :: weights
        type(c_ptr), intent(in) :: activation, output
        integer(c_size_t), intent(in) :: activation_elements, output_elements
        integer, intent(in) :: batch
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(weights%handle) .or. &
            .not. c_associated(activation) .or. .not. c_associated(output) .or. &
            activation_elements <= 0_c_size_t .or. output_elements <= 0_c_size_t .or. batch <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA device F32 batched matmul')
            return
        end if
        code = c_matmul_device_f32(context%handle, weights%handle, activation, activation_elements, &
            int(batch, c_int), output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA device F32 batched matmul failed')
    end subroutine cuda_q8_matmul_device_f32

    subroutine cuda_qwen35_embedding_device(context, weights, token_id, output, output_elements, stat)
        class(cuda_q8_context_t), intent(in) :: context
        class(cuda_q8_weights_t), intent(in) :: weights
        integer(c_int64_t), intent(in) :: token_id
        type(c_ptr), intent(in) :: output
        integer(c_size_t), intent(in) :: output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(weights%handle) .or. &
            .not. c_associated(output) .or. token_id < 0_c_int64_t .or. output_elements <= 0_c_size_t) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q8 embedding')
            return
        end if
        code = c_embedding_device(context%handle, weights%handle, token_id, output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q8 embedding lookup failed')
    end subroutine cuda_qwen35_embedding_device

    subroutine cuda_qwen35_embedding_device_batch(context, weights, host_tokens, batch, output, output_elements, stat)
        class(cuda_q8_context_t), intent(in) :: context
        class(cuda_q8_weights_t), intent(in) :: weights
        integer(c_int32_t), contiguous, target, intent(in) :: host_tokens(:)
        integer, intent(in) :: batch
        type(c_ptr), intent(in) :: output
        integer(c_size_t), intent(in) :: output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(weights%handle) .or. &
            .not. c_associated(output) .or. batch <= 0 .or. size(host_tokens) < batch .or. &
            output_elements <= 0_c_size_t) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA batched embedding arguments')
            return
        end if
        code = c_embedding_device_batch(context%handle, weights%handle, c_loc(host_tokens), &
            int(batch, c_int), output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA batched embedding lookup failed')
    end subroutine cuda_qwen35_embedding_device_batch

    subroutine cuda_qwen35_copy_device(context, device_input, device_output, bytes, stat)
        class(cuda_q8_context_t), intent(in) :: context
        type(c_ptr), intent(in) :: device_input, device_output
        integer(c_size_t), intent(in) :: bytes
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(device_input) .or. &
            .not. c_associated(device_output) .or. bytes <= 0_c_size_t) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA device copy')
            return
        end if
        code = c_qwen35_copy_device(context%handle, device_input, device_output, bytes)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA device copy failed')
    end subroutine cuda_qwen35_copy_device

    subroutine cuda_qwen35_add_device(context, device_left, device_right, device_output, elements, stat)
        class(cuda_q8_context_t), intent(in) :: context
        type(c_ptr), intent(in) :: device_left, device_right, device_output
        integer(c_size_t), intent(in) :: elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(device_left) .or. &
            .not. c_associated(device_right) .or. .not. c_associated(device_output) .or. &
            elements <= 0_c_size_t) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA device add')
            return
        end if
        code = c_qwen35_add_device(context%handle, device_left, device_right, device_output, elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA device add failed')
    end subroutine cuda_qwen35_add_device

    subroutine cuda_qwen35_copy_column_device(context, device_input, stride, column, device_output, elements, stat)
        class(cuda_q8_context_t), intent(in) :: context
        type(c_ptr), intent(in) :: device_input, device_output
        integer(c_size_t), intent(in) :: stride, elements
        integer, intent(in) :: column
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(device_input) .or. &
            .not. c_associated(device_output) .or. stride <= 0_c_size_t .or. elements <= 0_c_size_t .or. &
            column < 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA device column copy')
            return
        end if
        code = c_qwen35_copy_column_device(context%handle, device_input, stride, int(column, c_int), &
            device_output, elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA device column copy failed')
    end subroutine cuda_qwen35_copy_column_device

    subroutine cuda_qwen35_add_matrix_device(context, device_left, device_right, device_output, elements, stat)
        class(cuda_q8_context_t), intent(in) :: context
        type(c_ptr), intent(in) :: device_left, device_right, device_output
        integer(c_size_t), intent(in) :: elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(device_left) .or. &
            .not. c_associated(device_right) .or. .not. c_associated(device_output) .or. elements <= 0_c_size_t) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA matrix add arguments')
            return
        end if
        code = c_qwen35_add_matrix_device(context%handle, device_left, device_right, device_output, elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, 'CUDA matrix add failed')
    end subroutine cuda_qwen35_add_matrix_device

    subroutine cuda_qwen35_rms_norm_device(context, device_input, device_weights, device_output, &
            elements, epsilon, stat)
        class(cuda_q8_context_t), intent(in) :: context
        type(c_ptr), intent(in) :: device_input, device_weights, device_output
        integer(c_size_t), intent(in) :: elements
        real(c_float), intent(in) :: epsilon
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(device_input) .or. &
            .not. c_associated(device_weights) .or. .not. c_associated(device_output) .or. &
            elements <= 0_c_size_t .or. epsilon <= 0.0_c_float) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA device RMS norm')
            return
        end if
        code = c_qwen35_rms_norm_device(context%handle, device_input, device_weights, device_output, &
            elements, epsilon)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA device RMS norm failed')
    end subroutine cuda_qwen35_rms_norm_device

    subroutine cuda_qwen35_rms_norm_matrix_device(context, device_input, device_weights, device_output, &
            hidden, batch, epsilon, stat)
        class(cuda_q8_context_t), intent(in) :: context
        type(c_ptr), intent(in) :: device_input, device_weights, device_output
        integer(c_size_t), intent(in) :: hidden
        integer, intent(in) :: batch
        real(c_float), intent(in) :: epsilon
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(device_input) .or. &
            .not. c_associated(device_weights) .or. .not. c_associated(device_output) .or. &
            hidden <= 0_c_size_t .or. batch <= 0 .or. epsilon <= 0.0_c_float) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA batched RMS norm arguments')
            return
        end if
        code = c_qwen35_rms_norm_matrix_device(context%handle, device_input, device_weights, device_output, &
            hidden, int(batch, c_int), epsilon)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, 'CUDA batched RMS norm failed')
    end subroutine cuda_qwen35_rms_norm_matrix_device

    subroutine cuda_qwen35_silu_product_device(context, device_gate, device_up, elements, stat)
        class(cuda_q8_context_t), intent(in) :: context
        type(c_ptr), intent(in) :: device_gate, device_up
        integer(c_size_t), intent(in) :: elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(device_gate) .or. &
            .not. c_associated(device_up)) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA SiLU product arguments')
            return
        end if
        code = c_qwen35_silu_product_device(context%handle, device_gate, device_up, elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA SiLU product failed')
    end subroutine cuda_qwen35_silu_product_device

    subroutine cuda_qwen35_silu_product_matrix_device(context, device_gate, device_up, elements, stat)
        class(cuda_q8_context_t), intent(in) :: context
        type(c_ptr), intent(in) :: device_gate, device_up
        integer(c_size_t), intent(in) :: elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(device_gate) .or. &
            .not. c_associated(device_up) .or. elements <= 0_c_size_t) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA matrix SiLU arguments')
            return
        end if
        code = c_qwen35_silu_product_matrix_device(context%handle, device_gate, device_up, elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, 'CUDA matrix SiLU product failed')
    end subroutine cuda_qwen35_silu_product_matrix_device

    subroutine cuda_qwen35_argmax_device(context, device_logits, elements, host_index, stat)
        class(cuda_q8_context_t), intent(in) :: context
        type(c_ptr), intent(in) :: device_logits
        integer(c_size_t), intent(in) :: elements
        integer, intent(out) :: host_index
        type(status_t), intent(out) :: stat
        integer(c_int) :: code, index_c

        call stat%clear()
        host_index = 0
        if (.not. c_associated(context%handle) .or. .not. c_associated(device_logits) .or. &
            elements <= 0_c_size_t) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA device argmax')
            return
        end if
        code = c_qwen35_argmax_device(context%handle, device_logits, elements, index_c)
        if (code /= FORTAI_CUDA_OK) then
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA device argmax failed')
            return
        end if
        host_index = int(index_c)
    end subroutine cuda_qwen35_argmax_device

    subroutine cuda_q8_matvec_host(context, weights, activation, activation_bytes, output, &
            output_bytes, elapsed_ms, stat)
        class(cuda_q8_context_t), intent(in) :: context
        class(cuda_q8_weights_t), intent(in) :: weights
        integer(c_int8_t), contiguous, target, intent(in) :: activation(:)
        integer(c_size_t), intent(in) :: activation_bytes, output_bytes
        real(c_float), contiguous, target, intent(out) :: output(:)
        real(c_float), intent(out) :: elapsed_ms
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        elapsed_ms = 0.0_c_float
        if (.not. c_associated(context%handle) .or. .not. c_associated(weights%handle) .or. &
            size(activation) <= 0 .or. size(output) <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q8 host matvec')
            return
        end if
        code = c_matvec_host(context%handle, weights%handle, activation, activation_bytes, &
            output, output_bytes, elapsed_ms)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q8 host matvec failed')
    end subroutine cuda_q8_matvec_host

    subroutine cuda_q8_matvec_host_pair(context, first_weights, second_weights, activation, &
            activation_bytes, first_output, first_output_bytes, second_output, second_output_bytes, &
            elapsed_ms, stat)
        class(cuda_q8_context_t), intent(in) :: context
        class(cuda_q8_weights_t), intent(in) :: first_weights, second_weights
        integer(c_int8_t), contiguous, target, intent(in) :: activation(:)
        integer(c_size_t), intent(in) :: activation_bytes, first_output_bytes, second_output_bytes
        real(c_float), contiguous, target, intent(out) :: first_output(:), second_output(:)
        real(c_float), intent(out) :: elapsed_ms
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        elapsed_ms = 0.0_c_float
        if (.not. c_associated(context%handle) .or. .not. c_associated(first_weights%handle) .or. &
            .not. c_associated(second_weights%handle) .or. size(activation) <= 0 .or. &
            size(first_output) <= 0 .or. size(second_output) <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q8 host matvec pair')
            return
        end if
        code = c_matvec_host_pair(context%handle, first_weights%handle, second_weights%handle, &
            activation, activation_bytes, first_output, first_output_bytes, second_output, &
            second_output_bytes, elapsed_ms)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q8 host matvec pair failed')
    end subroutine cuda_q8_matvec_host_pair

    subroutine cuda_q8_matvec_host_triplet(context, first_weights, second_weights, third_weights, &
            activation, activation_bytes, first_output, first_output_bytes, second_output, &
            second_output_bytes, third_output, third_output_bytes, elapsed_ms, stat)
        class(cuda_q8_context_t), intent(in) :: context
        class(cuda_q8_weights_t), intent(in) :: first_weights, second_weights, third_weights
        integer(c_int8_t), contiguous, target, intent(in) :: activation(:)
        integer(c_size_t), intent(in) :: activation_bytes
        integer(c_size_t), intent(in) :: first_output_bytes, second_output_bytes, third_output_bytes
        real(c_float), contiguous, target, intent(out) :: first_output(:), second_output(:), third_output(:)
        real(c_float), intent(out) :: elapsed_ms
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        elapsed_ms = 0.0_c_float
        if (.not. c_associated(context%handle) .or. .not. c_associated(first_weights%handle) .or. &
            .not. c_associated(second_weights%handle) .or. .not. c_associated(third_weights%handle) .or. &
            size(activation) <= 0 .or. size(first_output) <= 0 .or. size(second_output) <= 0 .or. &
            size(third_output) <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q8 host matvec triplet')
            return
        end if
        code = c_matvec_host_triplet(context%handle, first_weights%handle, second_weights%handle, &
            third_weights%handle, activation, activation_bytes, first_output, first_output_bytes, &
            second_output, second_output_bytes, third_output, third_output_bytes, elapsed_ms)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q8 host matvec triplet failed')
    end subroutine cuda_q8_matvec_host_triplet

    subroutine cuda_q8_matvec_host_triplet_contiguous(context, first_weights, second_weights, &
            third_weights, activation, activation_bytes, host_output, host_output_bytes, elapsed_ms, stat)
        class(cuda_q8_context_t), intent(in) :: context
        class(cuda_q8_weights_t), intent(in) :: first_weights, second_weights, third_weights
        integer(c_int8_t), contiguous, target, intent(in) :: activation(:)
        integer(c_size_t), intent(in) :: activation_bytes, host_output_bytes
        real(c_float), contiguous, target, intent(out) :: host_output(:)
        real(c_float), intent(out) :: elapsed_ms
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        elapsed_ms = 0.0_c_float
        if (.not. c_associated(context%handle) .or. .not. c_associated(first_weights%handle) .or. &
            .not. c_associated(second_weights%handle) .or. .not. c_associated(third_weights%handle) .or. &
            size(activation) <= 0 .or. size(host_output) <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q8 contiguous triplet')
            return
        end if
        code = c_matvec_host_triplet_contiguous(context%handle, first_weights%handle, &
            second_weights%handle, third_weights%handle, activation, activation_bytes, host_output, &
            host_output_bytes, elapsed_ms)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q8 contiguous triplet failed')
    end subroutine cuda_q8_matvec_host_triplet_contiguous

    subroutine cuda_q8_ffn_host(context, gate_weights, up_weights, down_weights, host_activation, &
            activation_bytes, host_output, output_bytes, elapsed_ms, stat)
        class(cuda_q8_context_t), intent(in) :: context
        class(cuda_q8_weights_t), intent(in) :: gate_weights, up_weights, down_weights
        integer(c_int8_t), contiguous, target, intent(in) :: host_activation(:)
        integer(c_size_t), intent(in) :: activation_bytes, output_bytes
        real(c_float), contiguous, target, intent(out) :: host_output(:)
        real(c_float), intent(out) :: elapsed_ms
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        elapsed_ms = 0.0_c_float
        if (.not. c_associated(context%handle) .or. .not. c_associated(gate_weights%handle) .or. &
            .not. c_associated(up_weights%handle) .or. .not. c_associated(down_weights%handle) .or. &
            size(host_activation) <= 0 .or. size(host_output) <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q8 FFN host operation')
            return
        end if
        code = c_ffn_host(context%handle, gate_weights%handle, up_weights%handle, down_weights%handle, &
            host_activation, activation_bytes, host_output, output_bytes, elapsed_ms)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q8 FFN host operation failed')
    end subroutine cuda_q8_ffn_host

    subroutine cuda_q8_ffn_device(context, gate_weights, up_weights, down_weights, device_activation, &
            activation_elements, device_output, output_elements, stat)
        class(cuda_q8_context_t), intent(in) :: context
        class(cuda_q8_weights_t), intent(in) :: gate_weights, up_weights, down_weights
        type(c_ptr), intent(in) :: device_activation, device_output
        integer(c_size_t), intent(in) :: activation_elements, output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(gate_weights%handle) .or. &
            .not. c_associated(up_weights%handle) .or. .not. c_associated(down_weights%handle) .or. &
            .not. c_associated(device_activation) .or. .not. c_associated(device_output) .or. &
            activation_elements <= 0_c_size_t .or. output_elements <= 0_c_size_t) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Q8 FFN device operation')
            return
        end if
        code = c_ffn_device(context%handle, gate_weights%handle, up_weights%handle, down_weights%handle, &
            device_activation, activation_elements, device_output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Q8 FFN device operation failed')
    end subroutine cuda_q8_ffn_device

    subroutine cuda_qwen35_recurrent_create(self, context, qkv_weights, gate_weights, alpha_weights, &
            beta_weights, output_weights, conv_weights, conv_weight_bytes, conv_size, conv_kernel, &
            ssm_a, ssm_a_bytes, ssm_dt, ssm_dt_bytes, ssm_norm, ssm_norm_bytes, state_size, &
            key_heads, value_heads, head_size, inner_size, norm_epsilon, stat)
        class(cuda_qwen35_recurrent_t), intent(inout) :: self
        class(cuda_q8_context_t), intent(in) :: context
        class(cuda_q8_weights_t), intent(in) :: qkv_weights, gate_weights, alpha_weights
        class(cuda_q8_weights_t), intent(in) :: beta_weights, output_weights
        integer(c_int8_t), contiguous, target, intent(in) :: conv_weights(:), ssm_a(:), ssm_dt(:), ssm_norm(:)
        integer(c_size_t), intent(in) :: conv_weight_bytes, ssm_a_bytes, ssm_dt_bytes, ssm_norm_bytes
        integer, intent(in) :: conv_size, conv_kernel, state_size, key_heads, value_heads
        integer, intent(in) :: head_size, inner_size
        real(c_float), intent(in) :: norm_epsilon
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(qkv_weights%handle) .or. &
            .not. c_associated(gate_weights%handle) .or. .not. c_associated(alpha_weights%handle) .or. &
            .not. c_associated(beta_weights%handle) .or. .not. c_associated(output_weights%handle) .or. &
            size(conv_weights) <= 0 .or. size(ssm_a) <= 0 .or. size(ssm_dt) <= 0 .or. &
            size(ssm_norm) <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid Qwen3.5 CUDA recurrent layer')
            return
        end if
        if (c_associated(self%handle)) call self%destroy(stat)
        code = c_qwen35_recurrent_create(context%handle, qkv_weights%handle, gate_weights%handle, &
            alpha_weights%handle, beta_weights%handle, output_weights%handle, conv_weights, &
            conv_weight_bytes, int(conv_size, c_int), int(conv_kernel, c_int), ssm_a, ssm_a_bytes, &
            ssm_dt, ssm_dt_bytes, ssm_norm, ssm_norm_bytes, int(state_size, c_int), &
            int(key_heads, c_int), int(value_heads, c_int), int(head_size, c_int), int(inner_size, c_int), &
            norm_epsilon, self%handle)
        if (code /= FORTAI_CUDA_OK) then
            self%handle = c_null_ptr
            call stat%set(FORTAI_UNSUPPORTED, 'Qwen3.5 CUDA recurrent layer creation failed')
        end if
    end subroutine cuda_qwen35_recurrent_create

    subroutine cuda_qwen35_recurrent_create_state(self, context, conv_weights, conv_weight_bytes, conv_size, &
            conv_kernel, ssm_a, ssm_a_bytes, ssm_dt, ssm_dt_bytes, ssm_norm, ssm_norm_bytes, state_size, &
            key_heads, value_heads, head_size, inner_size, norm_epsilon, stat)
        class(cuda_qwen35_recurrent_t), intent(inout) :: self
        class(cuda_q8_context_t), intent(in) :: context
        integer(c_int8_t), contiguous, target, intent(in) :: conv_weights(:), ssm_a(:), ssm_dt(:), ssm_norm(:)
        integer(c_size_t), intent(in) :: conv_weight_bytes, ssm_a_bytes, ssm_dt_bytes, ssm_norm_bytes
        integer, intent(in) :: conv_size, conv_kernel, state_size, key_heads, value_heads, head_size, inner_size
        real(c_float), intent(in) :: norm_epsilon
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(context%handle) .or. size(conv_weights) <= 0 .or. size(ssm_a) <= 0 .or. &
            size(ssm_dt) <= 0 .or. size(ssm_norm) <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid Qwen3.5 CUDA recurrent state arguments')
            return
        end if
        if (c_associated(self%handle)) call self%destroy(stat)
        code = c_qwen35_recurrent_create_state(context%handle, conv_weights, conv_weight_bytes, &
            int(conv_size, c_int), int(conv_kernel, c_int), ssm_a, ssm_a_bytes, ssm_dt, ssm_dt_bytes, &
            ssm_norm, ssm_norm_bytes, int(state_size, c_int), int(key_heads, c_int), int(value_heads, c_int), &
            int(head_size, c_int), int(inner_size, c_int), norm_epsilon, self%handle)
        if (code /= FORTAI_CUDA_OK) then
            self%handle = c_null_ptr
            call stat%set(FORTAI_UNSUPPORTED, 'Qwen3.5 CUDA recurrent state creation failed')
        end if
    end subroutine cuda_qwen35_recurrent_create_state

    subroutine cuda_qwen35_recurrent_destroy(self, stat)
        class(cuda_qwen35_recurrent_t), intent(inout) :: self
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle)) return
        code = c_qwen35_recurrent_destroy(self%handle)
        self%handle = c_null_ptr
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'Qwen3.5 CUDA recurrent layer destruction failed')
    end subroutine cuda_qwen35_recurrent_destroy

    subroutine cuda_qwen35_recurrent_reset(self, stat)
        class(cuda_qwen35_recurrent_t), intent(in) :: self
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle)) return
        code = c_qwen35_recurrent_reset(self%handle)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'Qwen3.5 CUDA recurrent layer reset failed')
    end subroutine cuda_qwen35_recurrent_reset

    subroutine cuda_qwen35_recurrent_run(self, host_activation, activation_bytes, host_output, &
            output_bytes, elapsed_ms, stat)
        class(cuda_qwen35_recurrent_t), intent(in) :: self
        integer(c_int8_t), contiguous, target, intent(in) :: host_activation(:)
        integer(c_size_t), intent(in) :: activation_bytes, output_bytes
        real(c_float), contiguous, target, intent(out) :: host_output(:)
        real(c_float), intent(out) :: elapsed_ms
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        elapsed_ms = 0.0_c_float
        if (.not. c_associated(self%handle) .or. size(host_activation) <= 0 .or. &
            size(host_output) <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid Qwen3.5 CUDA recurrent run')
            return
        end if
        code = c_qwen35_recurrent_run(self%handle, host_activation, activation_bytes, host_output, &
            output_bytes, elapsed_ms)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'Qwen3.5 CUDA recurrent run failed')
    end subroutine cuda_qwen35_recurrent_run

    subroutine cuda_qwen35_recurrent_run_device(self, device_activation, activation_elements, &
            device_output, output_elements, stat)
        class(cuda_qwen35_recurrent_t), intent(in) :: self
        type(c_ptr), intent(in) :: device_activation, device_output
        integer(c_size_t), intent(in) :: activation_elements, output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle) .or. .not. c_associated(device_activation) .or. &
            .not. c_associated(device_output) .or. activation_elements <= 0_c_size_t .or. &
            output_elements <= 0_c_size_t) then
            call stat%set(FORTAI_INVALID, 'invalid Qwen3.5 CUDA recurrent device run')
            return
        end if
        code = c_qwen35_recurrent_run_device(self%handle, device_activation, activation_elements, &
            device_output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'Qwen3.5 CUDA recurrent device run failed')
    end subroutine cuda_qwen35_recurrent_run_device

    subroutine cuda_qwen35_recurrent_run_core_device(self, device_qkv, qkv_elements, device_gate, gate_elements, &
            device_alpha, alpha_elements, device_beta, beta_elements, device_output, output_elements, stat)
        class(cuda_qwen35_recurrent_t), intent(in) :: self
        type(c_ptr), intent(in) :: device_qkv, device_gate, device_alpha, device_beta, device_output
        integer(c_size_t), intent(in) :: qkv_elements, gate_elements, alpha_elements, beta_elements, output_elements
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle) .or. .not. c_associated(device_qkv) .or. &
            .not. c_associated(device_gate) .or. .not. c_associated(device_alpha) .or. &
            .not. c_associated(device_beta) .or. .not. c_associated(device_output)) then
            call stat%set(FORTAI_INVALID, 'invalid Qwen3.5 CUDA recurrent core arguments')
            return
        end if
        code = c_qwen35_recurrent_run_core_device(self%handle, device_qkv, qkv_elements, device_gate, gate_elements, &
            device_alpha, alpha_elements, device_beta, beta_elements, device_output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'Qwen3.5 CUDA recurrent core failed')
    end subroutine cuda_qwen35_recurrent_run_core_device

    subroutine cuda_qwen35_recurrent_run_core_device_batch(self, device_qkv, qkv_elements, device_gate, gate_elements, &
            device_alpha, alpha_elements, device_beta, beta_elements, batch, device_output, output_elements, stat)
        class(cuda_qwen35_recurrent_t), intent(in) :: self
        type(c_ptr), intent(in) :: device_qkv, device_gate, device_alpha, device_beta, device_output
        integer(c_size_t), intent(in) :: qkv_elements, gate_elements, alpha_elements, beta_elements, output_elements
        integer, intent(in) :: batch
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle) .or. .not. c_associated(device_qkv) .or. &
            .not. c_associated(device_gate) .or. .not. c_associated(device_alpha) .or. &
            .not. c_associated(device_beta) .or. .not. c_associated(device_output) .or. batch <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid Qwen3.5 CUDA recurrent batched core arguments')
            return
        end if
        code = c_qwen35_recurrent_run_core_device_batch(self%handle, device_qkv, qkv_elements, device_gate, gate_elements, &
            device_alpha, alpha_elements, device_beta, beta_elements, int(batch, c_int), device_output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'Qwen3.5 CUDA recurrent batched core failed')
    end subroutine cuda_qwen35_recurrent_run_core_device_batch

    subroutine cuda_qwen35_attention_create(self, context, query_weights, key_weights, value_weights, &
            output_weights, query_norm, query_norm_bytes, key_norm, key_norm_bytes, heads, &
            key_value_heads, head_size, value_size, max_context, rope_dimension, rope_base, &
            norm_epsilon, stat, cache_key_q8, cache_value_q8)
        class(cuda_qwen35_attention_t), intent(inout) :: self
        class(cuda_q8_context_t), intent(in) :: context
        class(cuda_q8_weights_t), intent(in) :: query_weights, key_weights, value_weights, output_weights
        integer(c_int8_t), contiguous, target, intent(in) :: query_norm(:), key_norm(:)
        integer(c_size_t), intent(in) :: query_norm_bytes, key_norm_bytes
        integer, intent(in) :: heads, key_value_heads, head_size, value_size, max_context, rope_dimension
        real(c_float), intent(in) :: rope_base, norm_epsilon
        type(status_t), intent(out) :: stat
        logical, intent(in), optional :: cache_key_q8, cache_value_q8
        integer(c_int) :: code, cache_key_q8_c, cache_value_q8_c

        call stat%clear()
        if (.not. c_associated(context%handle) .or. .not. c_associated(query_weights%handle) .or. &
            .not. c_associated(key_weights%handle) .or. .not. c_associated(value_weights%handle) .or. &
            .not. c_associated(output_weights%handle) .or. size(query_norm) <= 0 .or. size(key_norm) <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA attention creation arguments')
            return
        end if
        if (c_associated(self%handle)) call self%destroy(stat)
        cache_key_q8_c = 0_c_int
        cache_value_q8_c = 0_c_int
        if (present(cache_key_q8)) then
            if (cache_key_q8) cache_key_q8_c = 1_c_int
        end if
        if (present(cache_value_q8)) then
            if (cache_value_q8) cache_value_q8_c = 1_c_int
        end if
        code = c_qwen35_attention_create(context%handle, query_weights%handle, key_weights%handle, &
            value_weights%handle, output_weights%handle, query_norm, query_norm_bytes, key_norm, &
            key_norm_bytes, int(heads, c_int), int(key_value_heads, c_int), int(head_size, c_int), &
            int(value_size, c_int), int(max_context, c_int), int(rope_dimension, c_int), rope_base, &
            norm_epsilon, cache_key_q8_c, cache_value_q8_c, self%handle)
        if (code /= FORTAI_CUDA_OK) then
            self%handle = c_null_ptr
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA Qwen attention creation failed')
        end if
    end subroutine cuda_qwen35_attention_create

    subroutine cuda_qwen35_attention_create_state(self, context, query_norm, query_norm_bytes, key_norm, &
            key_norm_bytes, heads, key_value_heads, head_size, value_size, max_context, rope_dimension, &
            rope_base, norm_epsilon, stat, cache_key_q8, cache_value_q8)
        class(cuda_qwen35_attention_t), intent(inout) :: self
        class(cuda_q8_context_t), intent(in) :: context
        integer(c_int8_t), contiguous, target, intent(in) :: query_norm(:), key_norm(:)
        integer(c_size_t), intent(in) :: query_norm_bytes, key_norm_bytes
        integer, intent(in) :: heads, key_value_heads, head_size, value_size, max_context, rope_dimension
        real(c_float), intent(in) :: rope_base, norm_epsilon
        type(status_t), intent(out) :: stat
        logical, intent(in), optional :: cache_key_q8, cache_value_q8
        integer(c_int) :: code, cache_key_q8_c, cache_value_q8_c

        call stat%clear()
        if (.not. c_associated(context%handle) .or. size(query_norm) <= 0 .or. size(key_norm) <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA attention state arguments')
            return
        end if
        if (c_associated(self%handle)) call self%destroy(stat)
        cache_key_q8_c = 0_c_int
        cache_value_q8_c = 0_c_int
        if (present(cache_key_q8)) then
            if (cache_key_q8) cache_key_q8_c = 1_c_int
        end if
        if (present(cache_value_q8)) then
            if (cache_value_q8) cache_value_q8_c = 1_c_int
        end if
        code = c_qwen35_attention_create_state(context%handle, query_norm, query_norm_bytes, key_norm, &
            key_norm_bytes, int(heads, c_int), int(key_value_heads, c_int), int(head_size, c_int), &
            int(value_size, c_int), int(max_context, c_int), int(rope_dimension, c_int), rope_base, &
            norm_epsilon, cache_key_q8_c, cache_value_q8_c, self%handle)
        if (code /= FORTAI_CUDA_OK) then
            self%handle = c_null_ptr
            call stat%set(FORTAI_UNSUPPORTED, 'CUDA Qwen attention state creation failed')
        end if
    end subroutine cuda_qwen35_attention_create_state

    subroutine cuda_qwen35_attention_destroy(self, stat)
        class(cuda_qwen35_attention_t), intent(inout) :: self
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle)) return
        code = c_qwen35_attention_destroy(self%handle)
        self%handle = c_null_ptr
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Qwen attention destruction failed')
    end subroutine cuda_qwen35_attention_destroy

    subroutine cuda_qwen35_attention_reset(self, stat)
        class(cuda_qwen35_attention_t), intent(inout) :: self
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle)) return
        code = c_qwen35_attention_reset(self%handle)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Qwen attention reset failed')
    end subroutine cuda_qwen35_attention_reset

    subroutine cuda_qwen35_attention_run_device(self, device_activation, activation_elements, position, &
            device_output, output_elements, stat)
        class(cuda_qwen35_attention_t), intent(in) :: self
        type(c_ptr), intent(in) :: device_activation, device_output
        integer(c_size_t), intent(in) :: activation_elements, output_elements
        integer, intent(in) :: position
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle) .or. .not. c_associated(device_activation) .or. &
            .not. c_associated(device_output) .or. position < 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Qwen attention execution arguments')
            return
        end if
        code = c_qwen35_attention_run_device(self%handle, device_activation, activation_elements, &
            int(position, c_int), device_output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Qwen attention device execution failed')
    end subroutine cuda_qwen35_attention_run_device

    subroutine cuda_qwen35_attention_run_core_device(self, device_query, query_elements, device_key, key_elements, &
            device_value, value_elements, position, device_output, output_elements, stat)
        class(cuda_qwen35_attention_t), intent(in) :: self
        type(c_ptr), intent(in) :: device_query, device_key, device_value, device_output
        integer(c_size_t), intent(in) :: query_elements, key_elements, value_elements, output_elements
        integer, intent(in) :: position
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle) .or. .not. c_associated(device_query) .or. &
            .not. c_associated(device_key) .or. .not. c_associated(device_value) .or. &
            .not. c_associated(device_output) .or. position < 0) then
            call stat%set(FORTAI_INVALID, 'invalid CUDA Qwen attention core arguments')
            return
        end if
        code = c_qwen35_attention_run_core_device(self%handle, device_query, query_elements, device_key, &
            key_elements, device_value, value_elements, int(position, c_int), device_output, output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'CUDA Qwen attention core failed')
    end subroutine cuda_qwen35_attention_run_core_device

    subroutine cuda_qwen35_attention_run_core_device_batch(self, device_query, query_elements, device_key, key_elements, &
            device_value, value_elements, position, batch, device_output, output_elements, stat)
        class(cuda_qwen35_attention_t), intent(in) :: self
        type(c_ptr), intent(in) :: device_query, device_key, device_value, device_output
        integer(c_size_t), intent(in) :: query_elements, key_elements, value_elements, output_elements
        integer, intent(in) :: position, batch
        type(status_t), intent(out) :: stat
        integer(c_int) :: code

        call stat%clear()
        if (.not. c_associated(self%handle) .or. .not. c_associated(device_query) .or. &
            .not. c_associated(device_key) .or. .not. c_associated(device_value) .or. &
            .not. c_associated(device_output) .or. position < 0 .or. batch <= 0) then
            call stat%set(FORTAI_INVALID, 'invalid Qwen attention batched core arguments')
            return
        end if
        code = c_qwen35_attention_run_core_device_batch(self%handle, device_query, query_elements, device_key, &
            key_elements, device_value, value_elements, int(position, c_int), int(batch, c_int), device_output, &
            output_elements)
        if (code /= FORTAI_CUDA_OK) call stat%set(FORTAI_UNSUPPORTED, &
            'Qwen attention batched core failed')
    end subroutine cuda_qwen35_attention_run_core_device_batch

    function cuda_q8_last_error(self) result(message)
        class(cuda_q8_context_t), intent(in) :: self
        character(len=:), allocatable :: message

        ! The C ABI keeps the detailed message for native callers.  The
        ! Fortran binding returns a stable diagnostic until C-string decoding
        ! is added to the public error layer.
        if (c_associated(self%handle)) then
            message = 'CUDA Q8 backend operation failed'
        else
            message = 'CUDA Q8 context is not initialized'
        end if
    end function cuda_q8_last_error

end module fortai_backend_cuda
