module fortai_ggml
    !! Minimal, typed ISO-C view of GGML's tensor/backend primitives.
    !!
    !! This is deliberately not a model wrapper.  FortAI model modules build
    !! their own graphs and own all state; this module only exposes the
    !! low-level tensor and device ABI shared by native model implementations.
    use, intrinsic :: iso_c_binding, only: c_associated, c_bool, c_float, c_int, c_int64_t, c_ptr, c_size_t, c_null_ptr
    implicit none
    private

    integer(c_int), parameter, public :: GGML_STATUS_ALLOC_FAILED = -2_c_int
    integer(c_int), parameter, public :: GGML_STATUS_FAILED = -1_c_int
    integer(c_int), parameter, public :: GGML_STATUS_SUCCESS = 0_c_int
    integer(c_int), parameter, public :: GGML_STATUS_ABORTED = 1_c_int

    integer(c_int), parameter, public :: GGML_TYPE_F32 = 0_c_int
    integer(c_int), parameter, public :: GGML_TYPE_F16 = 1_c_int
    integer(c_int), parameter, public :: GGML_TYPE_Q4_0 = 2_c_int
    integer(c_int), parameter, public :: GGML_TYPE_Q4_1 = 3_c_int
    integer(c_int), parameter, public :: GGML_TYPE_Q5_0 = 6_c_int
    integer(c_int), parameter, public :: GGML_TYPE_Q5_1 = 7_c_int
    integer(c_int), parameter, public :: GGML_TYPE_Q8_0 = 8_c_int
    integer(c_int), parameter, public :: GGML_TYPE_Q8_1 = 9_c_int
    integer(c_int), parameter, public :: GGML_TYPE_Q2_K = 10_c_int
    integer(c_int), parameter, public :: GGML_TYPE_Q3_K = 11_c_int
    integer(c_int), parameter, public :: GGML_TYPE_Q4_K = 12_c_int
    integer(c_int), parameter, public :: GGML_TYPE_Q5_K = 13_c_int
    integer(c_int), parameter, public :: GGML_TYPE_Q6_K = 14_c_int
    integer(c_int), parameter, public :: GGML_TYPE_Q8_K = 15_c_int
    integer(c_int), parameter, public :: GGML_TYPE_I8 = 24_c_int
    integer(c_int), parameter, public :: GGML_TYPE_I16 = 25_c_int
    integer(c_int), parameter, public :: GGML_TYPE_I32 = 26_c_int
    integer(c_int), parameter, public :: GGML_TYPE_I64 = 27_c_int

    integer(c_int), parameter, public :: GGML_BACKEND_DEVICE_TYPE_CPU = 0_c_int
    integer(c_int), parameter, public :: GGML_BACKEND_DEVICE_TYPE_GPU = 1_c_int
    integer(c_int), parameter, public :: GGML_BACKEND_DEVICE_TYPE_IGPU = 2_c_int
    integer(c_int), parameter, public :: GGML_BACKEND_DEVICE_TYPE_ACCEL = 3_c_int
    integer(c_int), parameter, public :: GGML_BACKEND_DEVICE_TYPE_META = 4_c_int

    integer(c_int), parameter, public :: GGML_GLU_OP_REGLU = 0_c_int
    integer(c_int), parameter, public :: GGML_GLU_OP_GEGLU = 1_c_int
    integer(c_int), parameter, public :: GGML_GLU_OP_SWIGLU = 2_c_int
    integer(c_int), parameter, public :: GGML_GLU_OP_SWIGLU_OAI = 3_c_int
    integer(c_int), parameter, public :: GGML_GLU_OP_GEGLU_ERF = 4_c_int
    integer(c_int), parameter, public :: GGML_GLU_OP_GEGLU_QUICK = 5_c_int

    integer(c_int), parameter, public :: GGML_PREC_DEFAULT = 0_c_int
    integer(c_int), parameter, public :: GGML_PREC_F32 = 10_c_int

    type, bind(C), public :: ggml_init_params_t
        integer(c_size_t) :: mem_size
        type(c_ptr) :: mem_buffer
        logical(c_bool) :: no_alloc
    end type ggml_init_params_t

    public :: ggml_init_params
    public :: ggml_context_init
    public :: ggml_context_free
    public :: ggml_backend_device
    public :: ggml_backend_for_device
    public :: ggml_backend_for_cpu
    public :: ggml_tensor_new
    public :: ggml_tensor_new_1d
    public :: ggml_tensor_new_2d
    public :: ggml_tensor_new_3d
    public :: ggml_tensor_new_4d
    public :: ggml_tensor_dup
    public :: ggml_tensor_view
    public :: ggml_tensor_view_1d
    public :: ggml_tensor_view_2d
    public :: ggml_tensor_view_3d
    public :: ggml_tensor_view_4d
    public :: ggml_tensor_reshape_1d
    public :: ggml_tensor_reshape_2d
    public :: ggml_tensor_reshape_3d
    public :: ggml_tensor_reshape_4d
    public :: ggml_tensor_cont
    public :: ggml_tensor_cont_2d
    public :: ggml_tensor_cast
    public :: ggml_tensor_permute
    public :: ggml_tensor_transpose
    public :: ggml_tensor_add
    public :: ggml_tensor_mul
    public :: ggml_tensor_norm
    public :: ggml_tensor_gelu
    public :: ggml_tensor_silu
    public :: ggml_tensor_mul_mat
    public :: ggml_tensor_conv_1d_ph
    public :: ggml_tensor_flash_attn_ext
    public :: ggml_tensor_cpy
    public :: ggml_tensor_set
    public :: ggml_tensor_get_rows
    public :: ggml_tensor_soft_max_ext
    public :: ggml_tensor_diag_mask_inf
    public :: ggml_tensor_scale
    public :: ggml_tensor_repeat
    public :: ggml_tensor_glu_split
    public :: ggml_tensor_swiglu_oai
    public :: ggml_graph_new
    public :: ggml_graph_build
    public :: ggml_graph_expand
    public :: ggml_tensor_set_input
    public :: ggml_tensor_set_output
    public :: ggml_graph_compute
    public :: ggml_backend_alloc_context
    public :: ggml_backend_alloc_buffer
    public :: ggml_backend_free_buffer
    public :: ggml_backend_buffer_size
    public :: ggml_backend_set_tensor
    public :: ggml_backend_get_tensor
    public :: ggml_backend_tensor_copy
    public :: ggml_backend_buffer_clear
    public :: ggml_backend_synchronize
    public :: ggml_backend_free
    public :: ggml_backend_memory
    public :: ggml_backend_sched_new
    public :: ggml_backend_sched_free
    public :: ggml_backend_sched_alloc_graph
    public :: ggml_backend_sched_graph_compute
    public :: ggml_backend_sched_synchronize
    public :: ggml_backend_sched_reset
    public :: ggml_backend_sched_buffer_size
    public :: ggml_tensor_nbytes
    public :: ggml_tensor_nelements
    public :: ggml_tensor_type_size
    public :: ggml_tensor_block_size
    public :: ggml_tensor_element_size
    public :: ggml_tensor_data
    public :: ggml_tensor_overhead
    public :: ggml_time_us
    public :: ggml_graph_overhead

    interface
        function c_ggml_init(params) bind(C, name='ggml_init') result(context)
            import c_ptr, ggml_init_params_t
            type(ggml_init_params_t), value :: params
            type(c_ptr) :: context
        end function c_ggml_init

        subroutine c_ggml_free(context) bind(C, name='ggml_free')
            import c_ptr
            type(c_ptr), value :: context
        end subroutine c_ggml_free

        function c_ggml_new_tensor(context, value_type, rank, shape) bind(C, name='ggml_new_tensor') result(tensor)
            import c_int, c_int64_t, c_ptr
            type(c_ptr), value :: context
            integer(c_int), value :: value_type, rank
            integer(c_int64_t), intent(in) :: shape(*)
            type(c_ptr) :: tensor
        end function c_ggml_new_tensor

        function c_ggml_new_tensor_1d(context, value_type, ne0) bind(C, name='ggml_new_tensor_1d') result(tensor)
            import c_int, c_int64_t, c_ptr
            type(c_ptr), value :: context
            integer(c_int), value :: value_type
            integer(c_int64_t), value :: ne0
            type(c_ptr) :: tensor
        end function c_ggml_new_tensor_1d

        function c_ggml_new_tensor_2d(context, value_type, ne0, ne1) bind(C, name='ggml_new_tensor_2d') result(tensor)
            import c_int, c_int64_t, c_ptr
            type(c_ptr), value :: context
            integer(c_int), value :: value_type
            integer(c_int64_t), value :: ne0, ne1
            type(c_ptr) :: tensor
        end function c_ggml_new_tensor_2d

        function c_ggml_new_tensor_3d(context, value_type, ne0, ne1, ne2) bind(C, name='ggml_new_tensor_3d') result(tensor)
            import c_int, c_int64_t, c_ptr
            type(c_ptr), value :: context
            integer(c_int), value :: value_type
            integer(c_int64_t), value :: ne0, ne1, ne2
            type(c_ptr) :: tensor
        end function c_ggml_new_tensor_3d

        function c_ggml_new_tensor_4d(context, value_type, ne0, ne1, ne2, ne3) bind(C, name='ggml_new_tensor_4d') result(tensor)
            import c_int, c_int64_t, c_ptr
            type(c_ptr), value :: context
            integer(c_int), value :: value_type
            integer(c_int64_t), value :: ne0, ne1, ne2, ne3
            type(c_ptr) :: tensor
        end function c_ggml_new_tensor_4d

        function c_ggml_dup_tensor(context, tensor) bind(C, name='ggml_dup_tensor') result(result_tensor)
            import c_ptr
            type(c_ptr), value :: context, tensor
            type(c_ptr) :: result_tensor
        end function c_ggml_dup_tensor

        function c_ggml_view_tensor(context, tensor) bind(C, name='ggml_view_tensor') result(result_tensor)
            import c_ptr
            type(c_ptr), value :: context, tensor
            type(c_ptr) :: result_tensor
        end function c_ggml_view_tensor

        function c_ggml_view_1d(context, tensor, ne0, offset) bind(C, name='ggml_view_1d') result(result_tensor)
            import c_int64_t, c_ptr, c_size_t
            type(c_ptr), value :: context, tensor
            integer(c_int64_t), value :: ne0
            integer(c_size_t), value :: offset
            type(c_ptr) :: result_tensor
        end function c_ggml_view_1d

        function c_ggml_view_2d(context, tensor, ne0, ne1, nb1, offset) bind(C, name='ggml_view_2d') result(result_tensor)
            import c_int64_t, c_ptr, c_size_t
            type(c_ptr), value :: context, tensor
            integer(c_int64_t), value :: ne0, ne1
            integer(c_size_t), value :: nb1, offset
            type(c_ptr) :: result_tensor
        end function c_ggml_view_2d

        function c_ggml_view_3d(context, tensor, ne0, ne1, ne2, nb1, nb2, offset) bind(C, name='ggml_view_3d') result(result_tensor)
            import c_int64_t, c_ptr, c_size_t
            type(c_ptr), value :: context, tensor
            integer(c_int64_t), value :: ne0, ne1, ne2
            integer(c_size_t), value :: nb1, nb2, offset
            type(c_ptr) :: result_tensor
        end function c_ggml_view_3d

        function c_ggml_view_4d(context, tensor, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset) &
                bind(C, name='ggml_view_4d') result(result_tensor)
            import c_int64_t, c_ptr, c_size_t
            type(c_ptr), value :: context, tensor
            integer(c_int64_t), value :: ne0, ne1, ne2, ne3
            integer(c_size_t), value :: nb1, nb2, nb3, offset
            type(c_ptr) :: result_tensor
        end function c_ggml_view_4d

        function c_ggml_reshape_1d(context, tensor, ne0) bind(C, name='ggml_reshape_1d') result(result_tensor)
            import c_int64_t, c_ptr
            type(c_ptr), value :: context, tensor
            integer(c_int64_t), value :: ne0
            type(c_ptr) :: result_tensor
        end function c_ggml_reshape_1d

        function c_ggml_reshape_2d(context, tensor, ne0, ne1) bind(C, name='ggml_reshape_2d') result(result_tensor)
            import c_int64_t, c_ptr
            type(c_ptr), value :: context, tensor
            integer(c_int64_t), value :: ne0, ne1
            type(c_ptr) :: result_tensor
        end function c_ggml_reshape_2d

        function c_ggml_reshape_3d(context, tensor, ne0, ne1, ne2) bind(C, name='ggml_reshape_3d') result(result_tensor)
            import c_int64_t, c_ptr
            type(c_ptr), value :: context, tensor
            integer(c_int64_t), value :: ne0, ne1, ne2
            type(c_ptr) :: result_tensor
        end function c_ggml_reshape_3d

        function c_ggml_reshape_4d(context, tensor, ne0, ne1, ne2, ne3) bind(C, name='ggml_reshape_4d') result(result_tensor)
            import c_int64_t, c_ptr
            type(c_ptr), value :: context, tensor
            integer(c_int64_t), value :: ne0, ne1, ne2, ne3
            type(c_ptr) :: result_tensor
        end function c_ggml_reshape_4d

        function c_ggml_cont(context, tensor) bind(C, name='ggml_cont') result(result_tensor)
            import c_ptr
            type(c_ptr), value :: context, tensor
            type(c_ptr) :: result_tensor
        end function c_ggml_cont

        function c_ggml_cont_2d(context, tensor, ne0, ne1) bind(C, name='ggml_cont_2d') result(result_tensor)
            import c_int64_t, c_ptr
            type(c_ptr), value :: context, tensor
            integer(c_int64_t), value :: ne0, ne1
            type(c_ptr) :: result_tensor
        end function c_ggml_cont_2d

        function c_ggml_cast(context, tensor, value_type) bind(C, name='ggml_cast') result(result_tensor)
            import c_int, c_ptr
            type(c_ptr), value :: context, tensor
            integer(c_int), value :: value_type
            type(c_ptr) :: result_tensor
        end function c_ggml_cast

        function c_ggml_permute(context, tensor, axis0, axis1, axis2, axis3) bind(C, name='ggml_permute') result(result_tensor)
            import c_int, c_ptr
            type(c_ptr), value :: context, tensor
            integer(c_int), value :: axis0, axis1, axis2, axis3
            type(c_ptr) :: result_tensor
        end function c_ggml_permute

        function c_ggml_transpose(context, tensor) bind(C, name='ggml_transpose') result(result_tensor)
            import c_ptr
            type(c_ptr), value :: context, tensor
            type(c_ptr) :: result_tensor
        end function c_ggml_transpose

        function c_ggml_add(context, first, second) bind(C, name='ggml_add') result(result_tensor)
            import c_ptr
            type(c_ptr), value :: context, first, second
            type(c_ptr) :: result_tensor
        end function c_ggml_add

        function c_ggml_mul(context, first, second) bind(C, name='ggml_mul') result(result_tensor)
            import c_ptr
            type(c_ptr), value :: context, first, second
            type(c_ptr) :: result_tensor
        end function c_ggml_mul

        function c_ggml_norm(context, tensor, eps) bind(C, name='ggml_norm') result(result_tensor)
            import c_float, c_ptr
            type(c_ptr), value :: context, tensor
            real(c_float), value :: eps
            type(c_ptr) :: result_tensor
        end function c_ggml_norm

        function c_ggml_gelu(context, tensor) bind(C, name='ggml_gelu') result(result_tensor)
            import c_ptr
            type(c_ptr), value :: context, tensor
            type(c_ptr) :: result_tensor
        end function c_ggml_gelu

        function c_ggml_silu(context, tensor) bind(C, name='ggml_silu') result(result_tensor)
            import c_ptr
            type(c_ptr), value :: context, tensor
            type(c_ptr) :: result_tensor
        end function c_ggml_silu

        function c_ggml_mul_mat(context, first, second) bind(C, name='ggml_mul_mat') result(result_tensor)
            import c_ptr
            type(c_ptr), value :: context, first, second
            type(c_ptr) :: result_tensor
        end function c_ggml_mul_mat

        function c_ggml_conv_1d_ph(context, kernel, input, stride, dilation) bind(C, name='ggml_conv_1d_ph') result(result_tensor)
            import c_int, c_ptr
            type(c_ptr), value :: context, kernel, input
            integer(c_int), value :: stride, dilation
            type(c_ptr) :: result_tensor
        end function c_ggml_conv_1d_ph

        function c_ggml_flash_attn_ext(context, query, key, value, mask, scale, max_bias, logit_softcap) &
                bind(C, name='ggml_flash_attn_ext') result(result_tensor)
            import c_float, c_ptr
            type(c_ptr), value :: context, query, key, value, mask
            real(c_float), value :: scale, max_bias, logit_softcap
            type(c_ptr) :: result_tensor
        end function c_ggml_flash_attn_ext

        function c_ggml_cpy(context, source, destination) bind(C, name='ggml_cpy') result(result_tensor)
            import c_ptr
            type(c_ptr), value :: context, source, destination
            type(c_ptr) :: result_tensor
        end function c_ggml_cpy

        function c_ggml_set(context, destination, source, nb1, nb2, nb3, offset) bind(C, name='ggml_set') result(result_tensor)
            import c_ptr, c_size_t
            type(c_ptr), value :: context, destination, source
            integer(c_size_t), value :: nb1, nb2, nb3, offset
            type(c_ptr) :: result_tensor
        end function c_ggml_set

        function c_ggml_get_rows(context, data, indices) bind(C, name='ggml_get_rows') result(result_tensor)
            import c_ptr
            type(c_ptr), value :: context, data, indices
            type(c_ptr) :: result_tensor
        end function c_ggml_get_rows

        function c_ggml_soft_max_ext(context, tensor, mask, scale, max_bias) bind(C, name='ggml_soft_max_ext') result(result_tensor)
            import c_float, c_ptr
            type(c_ptr), value :: context, tensor, mask
            real(c_float), value :: scale, max_bias
            type(c_ptr) :: result_tensor
        end function c_ggml_soft_max_ext

        function c_ggml_diag_mask_inf(context, tensor, n_past) bind(C, name='ggml_diag_mask_inf') result(result_tensor)
            import c_int, c_ptr
            type(c_ptr), value :: context, tensor
            integer(c_int), value :: n_past
            type(c_ptr) :: result_tensor
        end function c_ggml_diag_mask_inf

        function c_ggml_scale(context, tensor, scale) bind(C, name='ggml_scale') result(result_tensor)
            import c_float, c_ptr
            type(c_ptr), value :: context, tensor
            real(c_float), value :: scale
            type(c_ptr) :: result_tensor
        end function c_ggml_scale

        function c_ggml_repeat(context, first, second) bind(C, name='ggml_repeat') result(result_tensor)
            import c_ptr
            type(c_ptr), value :: context, first, second
            type(c_ptr) :: result_tensor
        end function c_ggml_repeat

        function c_ggml_glu_split(context, first, second, op) bind(C, name='ggml_glu_split') result(result_tensor)
            import c_int, c_ptr
            type(c_ptr), value :: context, first, second
            integer(c_int), value :: op
            type(c_ptr) :: result_tensor
        end function c_ggml_glu_split

        function c_ggml_swiglu_oai(context, first, second, alpha, limit) bind(C, name='ggml_swiglu_oai') result(result_tensor)
            import c_float, c_ptr
            type(c_ptr), value :: context, first, second
            real(c_float), value :: alpha, limit
            type(c_ptr) :: result_tensor
        end function c_ggml_swiglu_oai

        function c_ggml_backend_dev_by_type(value_type) bind(C, name='ggml_backend_dev_by_type') result(device)
            import c_int, c_ptr
            integer(c_int), value :: value_type
            type(c_ptr) :: device
        end function c_ggml_backend_dev_by_type

        function c_ggml_backend_dev_count() bind(C, name='ggml_backend_dev_count') result(count)
            import c_size_t
            integer(c_size_t) :: count
        end function c_ggml_backend_dev_count

        function c_ggml_backend_dev_get(index) bind(C, name='ggml_backend_dev_get') result(device)
            import c_ptr, c_size_t
            integer(c_size_t), value :: index
            type(c_ptr) :: device
        end function c_ggml_backend_dev_get

        function c_ggml_backend_dev_type(device) bind(C, name='ggml_backend_dev_type') result(value_type)
            import c_int, c_ptr
            type(c_ptr), value :: device
            integer(c_int) :: value_type
        end function c_ggml_backend_dev_type

        function c_ggml_backend_dev_init(device, params) bind(C, name='ggml_backend_dev_init') result(backend)
            import c_ptr
            type(c_ptr), value :: device, params
            type(c_ptr) :: backend
        end function c_ggml_backend_dev_init

        function c_ggml_backend_init_by_type(value_type, params) bind(C, name='ggml_backend_init_by_type') result(backend)
            import c_int, c_ptr
            integer(c_int), value :: value_type
            type(c_ptr), value :: params
            type(c_ptr) :: backend
        end function c_ggml_backend_init_by_type

        function c_ggml_backend_get_default_buffer_type(backend) &
                bind(C, name='ggml_backend_get_default_buffer_type') result(buffer_type)
            import c_ptr
            type(c_ptr), value :: backend
            type(c_ptr) :: buffer_type
        end function c_ggml_backend_get_default_buffer_type

        function c_ggml_backend_alloc_ctx_tensors(context, backend) bind(C, name='ggml_backend_alloc_ctx_tensors') result(buffer)
            import c_ptr
            type(c_ptr), value :: context, backend
            type(c_ptr) :: buffer
        end function c_ggml_backend_alloc_ctx_tensors

        function c_ggml_backend_alloc_buffer(backend, size) bind(C, name='ggml_backend_alloc_buffer') result(buffer)
            import c_ptr, c_size_t
            type(c_ptr), value :: backend
            integer(c_size_t), value :: size
            type(c_ptr) :: buffer
        end function c_ggml_backend_alloc_buffer

        subroutine c_ggml_backend_buffer_free(buffer) bind(C, name='ggml_backend_buffer_free')
            import c_ptr
            type(c_ptr), value :: buffer
        end subroutine c_ggml_backend_buffer_free

        subroutine c_ggml_backend_buffer_clear(buffer, value) bind(C, name='ggml_backend_buffer_clear')
            import c_int, c_ptr
            type(c_ptr), value :: buffer
            integer(c_int), value :: value
        end subroutine c_ggml_backend_buffer_clear

        function c_ggml_backend_buffer_get_size(buffer) bind(C, name='ggml_backend_buffer_get_size') result(size)
            import c_ptr, c_size_t
            type(c_ptr), value :: buffer
            integer(c_size_t) :: size
        end function c_ggml_backend_buffer_get_size

        function c_ggml_backend_sched_new(backends, bufts, n_backends, graph_size, parallel, op_offload) &
                bind(C, name='ggml_backend_sched_new') result(sched)
            import c_bool, c_int, c_ptr, c_size_t
            type(c_ptr), intent(in) :: backends(*)
            type(c_ptr), value :: bufts
            integer(c_int), value :: n_backends
            integer(c_size_t), value :: graph_size
            logical(c_bool), value :: parallel, op_offload
            type(c_ptr) :: sched
        end function c_ggml_backend_sched_new

        subroutine c_ggml_backend_sched_free(sched) bind(C, name='ggml_backend_sched_free')
            import c_ptr
            type(c_ptr), value :: sched
        end subroutine c_ggml_backend_sched_free

        function c_ggml_backend_sched_alloc_graph(sched, graph) bind(C, name='ggml_backend_sched_alloc_graph') result(ok)
            import c_bool, c_ptr
            type(c_ptr), value :: sched, graph
            logical(c_bool) :: ok
        end function c_ggml_backend_sched_alloc_graph

        function c_ggml_backend_sched_graph_compute(sched, graph) bind(C, name='ggml_backend_sched_graph_compute') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: sched, graph
            integer(c_int) :: code
        end function c_ggml_backend_sched_graph_compute

        subroutine c_ggml_backend_sched_synchronize(sched) bind(C, name='ggml_backend_sched_synchronize')
            import c_ptr
            type(c_ptr), value :: sched
        end subroutine c_ggml_backend_sched_synchronize

        subroutine c_ggml_backend_sched_reset(sched) bind(C, name='ggml_backend_sched_reset')
            import c_ptr
            type(c_ptr), value :: sched
        end subroutine c_ggml_backend_sched_reset

        function c_ggml_backend_sched_get_buffer_size(sched, backend) &
                bind(C, name='ggml_backend_sched_get_buffer_size') result(bytes)
            import c_ptr, c_size_t
            type(c_ptr), value :: sched, backend
            integer(c_size_t) :: bytes
        end function c_ggml_backend_sched_get_buffer_size

        function c_ggml_backend_buffer_is_host(buffer) bind(C, name='ggml_backend_buffer_is_host') result(is_host)
            import c_bool, c_ptr
            type(c_ptr), value :: buffer
            logical(c_bool) :: is_host
        end function c_ggml_backend_buffer_is_host

        subroutine c_ggml_backend_tensor_set(tensor, data, offset, size) bind(C, name='ggml_backend_tensor_set')
            import c_ptr, c_size_t
            type(c_ptr), value :: tensor, data
            integer(c_size_t), value :: offset, size
        end subroutine c_ggml_backend_tensor_set

        subroutine c_ggml_backend_tensor_get(tensor, data, offset, size) bind(C, name='ggml_backend_tensor_get')
            import c_ptr, c_size_t
            type(c_ptr), value :: tensor, data
            integer(c_size_t), value :: offset, size
        end subroutine c_ggml_backend_tensor_get

        subroutine c_ggml_backend_tensor_copy(source, destination) bind(C, name='ggml_backend_tensor_copy')
            import c_ptr
            type(c_ptr), value :: source, destination
        end subroutine c_ggml_backend_tensor_copy

        subroutine c_ggml_backend_synchronize(backend) bind(C, name='ggml_backend_synchronize')
            import c_ptr
            type(c_ptr), value :: backend
        end subroutine c_ggml_backend_synchronize

        subroutine c_ggml_backend_free(backend) bind(C, name='ggml_backend_free')
            import c_ptr
            type(c_ptr), value :: backend
        end subroutine c_ggml_backend_free

        subroutine c_ggml_backend_dev_memory(device, free_bytes, total_bytes) bind(C, name='ggml_backend_dev_memory')
            import c_ptr, c_size_t
            type(c_ptr), value :: device
            integer(c_size_t) :: free_bytes, total_bytes
        end subroutine c_ggml_backend_dev_memory

        function c_ggml_backend_graph_compute(backend, graph) bind(C, name='ggml_backend_graph_compute') result(code)
            import c_int, c_ptr
            type(c_ptr), value :: backend, graph
            integer(c_int) :: code
        end function c_ggml_backend_graph_compute

        function c_ggml_new_graph_custom(context, size, grads) bind(C, name='ggml_new_graph_custom') result(graph)
            import c_bool, c_ptr, c_size_t
            type(c_ptr), value :: context
            integer(c_size_t), value :: size
            logical(c_bool), value :: grads
            type(c_ptr) :: graph
        end function c_ggml_new_graph_custom

        subroutine c_ggml_build_forward_expand(graph, tensor) bind(C, name='ggml_build_forward_expand')
            import c_ptr
            type(c_ptr), value :: graph, tensor
        end subroutine c_ggml_build_forward_expand

        subroutine c_ggml_set_input(tensor) bind(C, name='ggml_set_input')
            import c_ptr
            type(c_ptr), value :: tensor
        end subroutine c_ggml_set_input

        subroutine c_ggml_set_output(tensor) bind(C, name='ggml_set_output')
            import c_ptr
            type(c_ptr), value :: tensor
        end subroutine c_ggml_set_output

        function c_ggml_nelements(tensor) bind(C, name='ggml_nelements') result(count)
            import c_int64_t, c_ptr
            type(c_ptr), value :: tensor
            integer(c_int64_t) :: count
        end function c_ggml_nelements

        function c_ggml_nbytes(tensor) bind(C, name='ggml_nbytes') result(bytes)
            import c_ptr, c_size_t
            type(c_ptr), value :: tensor
            integer(c_size_t) :: bytes
        end function c_ggml_nbytes

        function c_ggml_type_size(value_type) bind(C, name='ggml_type_size') result(bytes)
            import c_int, c_size_t
            integer(c_int), value :: value_type
            integer(c_size_t) :: bytes
        end function c_ggml_type_size

        function c_ggml_blck_size(value_type) bind(C, name='ggml_blck_size') result(block_size)
            import c_int, c_int64_t
            integer(c_int), value :: value_type
            integer(c_int64_t) :: block_size
        end function c_ggml_blck_size

        function c_ggml_element_size(tensor) bind(C, name='ggml_element_size') result(bytes)
            import c_ptr, c_size_t
            type(c_ptr), value :: tensor
            integer(c_size_t) :: bytes
        end function c_ggml_element_size

        function c_ggml_get_data(tensor) bind(C, name='ggml_get_data') result(data)
            import c_ptr
            type(c_ptr), value :: tensor
            type(c_ptr) :: data
        end function c_ggml_get_data

        function c_ggml_time_us() bind(C, name='ggml_time_us') result(time)
            import c_int64_t
            integer(c_int64_t) :: time
        end function c_ggml_time_us

        function c_ggml_tensor_overhead() bind(C, name='ggml_tensor_overhead') result(bytes)
            import c_size_t
            integer(c_size_t) :: bytes
        end function c_ggml_tensor_overhead


        function c_ggml_graph_overhead() bind(C, name='ggml_graph_overhead') result(bytes)
            import c_size_t
            integer(c_size_t) :: bytes
        end function c_ggml_graph_overhead

    end interface

contains

    function ggml_init_params(memory_size, no_alloc) result(params)
        integer(c_size_t), intent(in) :: memory_size
        logical(c_bool), intent(in), optional :: no_alloc
        type(ggml_init_params_t) :: params

        params%mem_size = memory_size
        params%mem_buffer = c_null_ptr
        params%no_alloc = .false._c_bool
        if (present(no_alloc)) params%no_alloc = no_alloc
    end function ggml_init_params

    function ggml_context_init(params) result(context)
        type(ggml_init_params_t), intent(in) :: params
        type(c_ptr) :: context

        context = c_ggml_init(params)
    end function ggml_context_init

    subroutine ggml_context_free(context)
        type(c_ptr), intent(inout) :: context

        if (c_associated(context)) call c_ggml_free(context)
        context = c_null_ptr
    end subroutine ggml_context_free

    function ggml_backend_device(device_type, ordinal) result(device)
        integer(c_int), intent(in) :: device_type
        integer(c_int), intent(in), optional :: ordinal
        type(c_ptr) :: device
        integer(c_size_t) :: i, count
        integer(c_int) :: wanted, current

        device = c_null_ptr
        wanted = 0_c_int
        if (present(ordinal)) wanted = max(0_c_int, ordinal)
        count = c_ggml_backend_dev_count()
        if (device_type == GGML_BACKEND_DEVICE_TYPE_CPU) then
            device = c_ggml_backend_dev_by_type(device_type)
            return
        end if
        if (count == 0_c_size_t) return
        current = 0_c_int
        do i = 0_c_size_t, count - 1_c_size_t
            device = c_ggml_backend_dev_get(i)
            if (c_associated(device)) then
                if (c_ggml_backend_dev_type(device) == device_type) then
                    if (current == wanted) return
                    current = current + 1_c_int
                end if
            end if
        end do
        device = c_null_ptr
    end function ggml_backend_device

    function ggml_backend_for_device(device) result(backend)
        type(c_ptr), intent(in) :: device
        type(c_ptr) :: backend

        backend = c_null_ptr
        if (c_associated(device)) backend = c_ggml_backend_dev_init(device, c_null_ptr)
    end function ggml_backend_for_device

    function ggml_backend_for_cpu() result(backend)
        type(c_ptr) :: backend

        backend = c_ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, c_null_ptr)
    end function ggml_backend_for_cpu

    function ggml_tensor_new(context, value_type, shape) result(tensor)
        type(c_ptr), intent(in) :: context
        integer(c_int), intent(in) :: value_type
        integer(c_int64_t), intent(in), contiguous :: shape(:)
        type(c_ptr) :: tensor

        tensor = c_ggml_new_tensor(context, value_type, int(size(shape), c_int), shape)
    end function ggml_tensor_new

    function ggml_tensor_new_1d(context, value_type, ne0) result(tensor)
        type(c_ptr), intent(in) :: context
        integer(c_int), intent(in) :: value_type
        integer(c_int64_t), intent(in) :: ne0
        type(c_ptr) :: tensor

        tensor = c_ggml_new_tensor_1d(context, value_type, ne0)
    end function ggml_tensor_new_1d

    function ggml_tensor_new_2d(context, value_type, ne0, ne1) result(tensor)
        type(c_ptr), intent(in) :: context
        integer(c_int), intent(in) :: value_type
        integer(c_int64_t), intent(in) :: ne0, ne1
        type(c_ptr) :: tensor

        tensor = c_ggml_new_tensor_2d(context, value_type, ne0, ne1)
    end function ggml_tensor_new_2d

    function ggml_tensor_new_3d(context, value_type, ne0, ne1, ne2) result(tensor)
        type(c_ptr), intent(in) :: context
        integer(c_int), intent(in) :: value_type
        integer(c_int64_t), intent(in) :: ne0, ne1, ne2
        type(c_ptr) :: tensor

        tensor = c_ggml_new_tensor_3d(context, value_type, ne0, ne1, ne2)
    end function ggml_tensor_new_3d

    function ggml_tensor_new_4d(context, value_type, ne0, ne1, ne2, ne3) result(tensor)
        type(c_ptr), intent(in) :: context
        integer(c_int), intent(in) :: value_type
        integer(c_int64_t), intent(in) :: ne0, ne1, ne2, ne3
        type(c_ptr) :: tensor

        tensor = c_ggml_new_tensor_4d(context, value_type, ne0, ne1, ne2, ne3)
    end function ggml_tensor_new_4d

    function ggml_tensor_dup(context, tensor) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_dup_tensor(context, tensor)
    end function ggml_tensor_dup

    function ggml_tensor_view(context, tensor) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_view_tensor(context, tensor)
    end function ggml_tensor_view

    function ggml_tensor_view_1d(context, tensor, ne0, offset) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        integer(c_int64_t), intent(in) :: ne0
        integer(c_size_t), intent(in) :: offset
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_view_1d(context, tensor, ne0, offset)
    end function ggml_tensor_view_1d

    function ggml_tensor_view_2d(context, tensor, ne0, ne1, nb1, offset) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        integer(c_int64_t), intent(in) :: ne0, ne1
        integer(c_size_t), intent(in) :: nb1, offset
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_view_2d(context, tensor, ne0, ne1, nb1, offset)
    end function ggml_tensor_view_2d

    function ggml_tensor_view_3d(context, tensor, ne0, ne1, ne2, nb1, nb2, offset) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        integer(c_int64_t), intent(in) :: ne0, ne1, ne2
        integer(c_size_t), intent(in) :: nb1, nb2, offset
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_view_3d(context, tensor, ne0, ne1, ne2, nb1, nb2, offset)
    end function ggml_tensor_view_3d

    function ggml_tensor_view_4d(context, tensor, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        integer(c_int64_t), intent(in) :: ne0, ne1, ne2, ne3
        integer(c_size_t), intent(in) :: nb1, nb2, nb3, offset
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_view_4d(context, tensor, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset)
    end function ggml_tensor_view_4d

    function ggml_tensor_reshape_1d(context, tensor, ne0) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        integer(c_int64_t), intent(in) :: ne0
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_reshape_1d(context, tensor, ne0)
    end function ggml_tensor_reshape_1d

    function ggml_tensor_reshape_2d(context, tensor, ne0, ne1) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        integer(c_int64_t), intent(in) :: ne0, ne1
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_reshape_2d(context, tensor, ne0, ne1)
    end function ggml_tensor_reshape_2d

    function ggml_tensor_reshape_3d(context, tensor, ne0, ne1, ne2) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        integer(c_int64_t), intent(in) :: ne0, ne1, ne2
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_reshape_3d(context, tensor, ne0, ne1, ne2)
    end function ggml_tensor_reshape_3d

    function ggml_tensor_reshape_4d(context, tensor, ne0, ne1, ne2, ne3) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        integer(c_int64_t), intent(in) :: ne0, ne1, ne2, ne3
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_reshape_4d(context, tensor, ne0, ne1, ne2, ne3)
    end function ggml_tensor_reshape_4d

    function ggml_tensor_cont(context, tensor) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_cont(context, tensor)
    end function ggml_tensor_cont

    function ggml_tensor_cont_2d(context, tensor, ne0, ne1) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        integer(c_int64_t), intent(in) :: ne0, ne1
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_cont_2d(context, tensor, ne0, ne1)
    end function ggml_tensor_cont_2d

    function ggml_tensor_cast(context, tensor, value_type) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        integer(c_int), intent(in) :: value_type
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_cast(context, tensor, value_type)
    end function ggml_tensor_cast

    function ggml_tensor_permute(context, tensor, axis0, axis1, axis2, axis3) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        integer(c_int), intent(in) :: axis0, axis1, axis2, axis3
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_permute(context, tensor, axis0, axis1, axis2, axis3)
    end function ggml_tensor_permute

    function ggml_tensor_transpose(context, tensor) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_transpose(context, tensor)
    end function ggml_tensor_transpose

    function ggml_tensor_add(context, first, second) result(result_tensor)
        type(c_ptr), intent(in) :: context, first, second
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_add(context, first, second)
    end function ggml_tensor_add

    function ggml_tensor_mul(context, first, second) result(result_tensor)
        type(c_ptr), intent(in) :: context, first, second
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_mul(context, first, second)
    end function ggml_tensor_mul

    function ggml_tensor_norm(context, tensor, eps) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        real, intent(in) :: eps
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_norm(context, tensor, real(eps, kind=kind(0.0)))
    end function ggml_tensor_norm

    function ggml_tensor_gelu(context, tensor) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_gelu(context, tensor)
    end function ggml_tensor_gelu

    function ggml_tensor_silu(context, tensor) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_silu(context, tensor)
    end function ggml_tensor_silu

    function ggml_tensor_mul_mat(context, first, second) result(result_tensor)
        type(c_ptr), intent(in) :: context, first, second
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_mul_mat(context, first, second)
    end function ggml_tensor_mul_mat

    function ggml_tensor_conv_1d_ph(context, kernel, input, stride, dilation) result(result_tensor)
        type(c_ptr), intent(in) :: context, kernel, input
        integer(c_int), intent(in) :: stride, dilation
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_conv_1d_ph(context, kernel, input, stride, dilation)
    end function ggml_tensor_conv_1d_ph

    function ggml_tensor_flash_attn_ext(context, query, key, value, mask, scale, max_bias, logit_softcap) result(result_tensor)
        type(c_ptr), intent(in) :: context, query, key, value, mask
        real, intent(in) :: scale, max_bias, logit_softcap
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_flash_attn_ext(context, query, key, value, mask, real(scale, kind=kind(0.0)), &
            real(max_bias, kind=kind(0.0)), real(logit_softcap, kind=kind(0.0)))
    end function ggml_tensor_flash_attn_ext

    function ggml_tensor_cpy(context, source, destination) result(result_tensor)
        type(c_ptr), intent(in) :: context, source, destination
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_cpy(context, source, destination)
    end function ggml_tensor_cpy

    function ggml_tensor_set(context, destination, source, nb1, nb2, nb3, offset) result(result_tensor)
        type(c_ptr), intent(in) :: context, destination, source
        integer(c_size_t), intent(in) :: nb1, nb2, nb3, offset
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_set(context, destination, source, nb1, nb2, nb3, offset)
    end function ggml_tensor_set

    function ggml_tensor_get_rows(context, data, indices) result(result_tensor)
        type(c_ptr), intent(in) :: context, data, indices
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_get_rows(context, data, indices)
    end function ggml_tensor_get_rows

    function ggml_tensor_soft_max_ext(context, tensor, mask, scale, max_bias) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor, mask
        real, intent(in) :: scale, max_bias
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_soft_max_ext(context, tensor, mask, real(scale, kind=kind(0.0)), &
            real(max_bias, kind=kind(0.0)))
    end function ggml_tensor_soft_max_ext

    function ggml_tensor_diag_mask_inf(context, tensor, n_past) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        integer(c_int), intent(in) :: n_past
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_diag_mask_inf(context, tensor, n_past)
    end function ggml_tensor_diag_mask_inf

    function ggml_tensor_scale(context, tensor, scale) result(result_tensor)
        type(c_ptr), intent(in) :: context, tensor
        real, intent(in) :: scale
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_scale(context, tensor, real(scale, kind=kind(0.0)))
    end function ggml_tensor_scale

    function ggml_tensor_repeat(context, first, second) result(result_tensor)
        type(c_ptr), intent(in) :: context, first, second
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_repeat(context, first, second)
    end function ggml_tensor_repeat

    function ggml_tensor_glu_split(context, first, second, op) result(result_tensor)
        type(c_ptr), intent(in) :: context, first, second
        integer(c_int), intent(in) :: op
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_glu_split(context, first, second, op)
    end function ggml_tensor_glu_split

    function ggml_tensor_swiglu_oai(context, first, second, alpha, limit) result(result_tensor)
        type(c_ptr), intent(in) :: context, first, second
        real, intent(in) :: alpha, limit
        type(c_ptr) :: result_tensor

        result_tensor = c_ggml_swiglu_oai(context, first, second, real(alpha, kind=kind(0.0)), &
            real(limit, kind=kind(0.0)))
    end function ggml_tensor_swiglu_oai

    function ggml_graph_new(context, size, grads) result(graph)
        type(c_ptr), intent(in) :: context
        integer(c_size_t), intent(in), optional :: size
        logical(c_bool), intent(in), optional :: grads
        type(c_ptr) :: graph
        integer(c_size_t) :: graph_size
        logical(c_bool) :: with_grads

        graph_size = 8192_c_size_t
        if (present(size)) graph_size = size
        with_grads = .false._c_bool
        if (present(grads)) with_grads = grads
        graph = c_ggml_new_graph_custom(context, graph_size, with_grads)
    end function ggml_graph_new

    subroutine ggml_graph_build(graph, output)
        type(c_ptr), intent(in) :: graph, output

        call c_ggml_build_forward_expand(graph, output)
    end subroutine ggml_graph_build

    subroutine ggml_graph_expand(graph, tensor)
        type(c_ptr), intent(in) :: graph, tensor

        call c_ggml_build_forward_expand(graph, tensor)
    end subroutine ggml_graph_expand

    subroutine ggml_tensor_set_input(tensor)
        type(c_ptr), intent(in) :: tensor

        call c_ggml_set_input(tensor)
    end subroutine ggml_tensor_set_input

    subroutine ggml_tensor_set_output(tensor)
        type(c_ptr), intent(in) :: tensor

        call c_ggml_set_output(tensor)
    end subroutine ggml_tensor_set_output

    integer(c_int) function ggml_graph_compute(backend, graph)
        type(c_ptr), intent(in) :: backend, graph

        ggml_graph_compute = c_ggml_backend_graph_compute(backend, graph)
    end function ggml_graph_compute

    function ggml_backend_alloc_context(context, backend) result(buffer)
        type(c_ptr), intent(in) :: context, backend
        type(c_ptr) :: buffer

        buffer = c_ggml_backend_alloc_ctx_tensors(context, backend)
    end function ggml_backend_alloc_context

    function ggml_backend_alloc_buffer(backend, bytes) result(buffer)
        type(c_ptr), intent(in) :: backend
        integer(c_size_t), intent(in) :: bytes
        type(c_ptr) :: buffer

        buffer = c_ggml_backend_alloc_buffer(backend, bytes)
    end function ggml_backend_alloc_buffer

    function ggml_backend_sched_new(backend, graph_size, parallel, op_offload, fallback) result(sched)
        type(c_ptr), intent(in) :: backend
        integer(c_size_t), intent(in), optional :: graph_size
        logical(c_bool), intent(in), optional :: parallel, op_offload
        type(c_ptr), intent(in), optional :: fallback
        type(c_ptr) :: sched
        type(c_ptr), target :: backends(2)
        integer(c_int) :: n_backends
        integer(c_size_t) :: actual_graph_size
        logical(c_bool) :: actual_parallel, actual_op_offload

        actual_graph_size = 4096_c_size_t
        if (present(graph_size)) actual_graph_size = graph_size
        actual_parallel = .false._c_bool
        if (present(parallel)) actual_parallel = parallel
        actual_op_offload = .true._c_bool
        if (present(op_offload)) actual_op_offload = op_offload
        backends(1) = backend
        n_backends = 1_c_int
        if (present(fallback)) then
            if (c_associated(fallback)) then
                backends(2) = fallback
                n_backends = 2_c_int
            end if
        end if
        sched = c_ggml_backend_sched_new(backends, c_null_ptr, n_backends, actual_graph_size, &
            actual_parallel, actual_op_offload)
    end function ggml_backend_sched_new

    subroutine ggml_backend_sched_free(sched)
        type(c_ptr), intent(inout) :: sched

        if (c_associated(sched)) call c_ggml_backend_sched_free(sched)
        sched = c_null_ptr
    end subroutine ggml_backend_sched_free

    logical function ggml_backend_sched_alloc_graph(sched, graph)
        type(c_ptr), intent(in) :: sched, graph

        ggml_backend_sched_alloc_graph = .false.
        if (c_associated(sched) .and. c_associated(graph)) then
            ggml_backend_sched_alloc_graph = c_ggml_backend_sched_alloc_graph(sched, graph)
        end if
    end function ggml_backend_sched_alloc_graph

    integer(c_int) function ggml_backend_sched_graph_compute(sched, graph)
        type(c_ptr), intent(in) :: sched, graph

        ggml_backend_sched_graph_compute = GGML_STATUS_FAILED
        if (c_associated(sched) .and. c_associated(graph)) then
            ggml_backend_sched_graph_compute = c_ggml_backend_sched_graph_compute(sched, graph)
        end if
    end function ggml_backend_sched_graph_compute

    subroutine ggml_backend_sched_synchronize(sched)
        type(c_ptr), intent(in) :: sched

        if (c_associated(sched)) call c_ggml_backend_sched_synchronize(sched)
    end subroutine ggml_backend_sched_synchronize

    subroutine ggml_backend_sched_reset(sched)
        type(c_ptr), intent(in) :: sched

        if (c_associated(sched)) call c_ggml_backend_sched_reset(sched)
    end subroutine ggml_backend_sched_reset

    integer(c_size_t) function ggml_backend_sched_buffer_size(sched, backend)
        type(c_ptr), intent(in) :: sched, backend

        ggml_backend_sched_buffer_size = 0_c_size_t
        if (c_associated(sched) .and. c_associated(backend)) then
            ggml_backend_sched_buffer_size = c_ggml_backend_sched_get_buffer_size(sched, backend)
        end if
    end function ggml_backend_sched_buffer_size

    integer(c_size_t) function ggml_backend_buffer_size(buffer)
        type(c_ptr), intent(in) :: buffer

        if (c_associated(buffer)) then
            ggml_backend_buffer_size = c_ggml_backend_buffer_get_size(buffer)
        else
            ggml_backend_buffer_size = 0_c_size_t
        end if
    end function ggml_backend_buffer_size

    subroutine ggml_backend_free_buffer(buffer)
        type(c_ptr), intent(inout) :: buffer

        if (c_associated(buffer)) call c_ggml_backend_buffer_free(buffer)
        buffer = c_null_ptr
    end subroutine ggml_backend_free_buffer

    subroutine ggml_backend_buffer_clear(buffer, value)
        type(c_ptr), intent(in) :: buffer
        integer, intent(in), optional :: value
        integer(c_int) :: actual_value

        actual_value = 0_c_int
        if (present(value)) actual_value = int(value, c_int)
        if (c_associated(buffer)) call c_ggml_backend_buffer_clear(buffer, actual_value)
    end subroutine ggml_backend_buffer_clear

    subroutine ggml_backend_set_tensor(tensor, data, offset, bytes)
        type(c_ptr), intent(in) :: tensor, data
        integer(c_size_t), intent(in), optional :: offset, bytes
        integer(c_size_t) :: actual_offset, actual_bytes

        actual_offset = 0_c_size_t
        if (present(offset)) actual_offset = offset
        actual_bytes = ggml_tensor_nbytes(tensor)
        if (present(bytes)) actual_bytes = bytes
        call c_ggml_backend_tensor_set(tensor, data, actual_offset, actual_bytes)
    end subroutine ggml_backend_set_tensor

    subroutine ggml_backend_get_tensor(tensor, data, offset, bytes)
        type(c_ptr), intent(in) :: tensor, data
        integer(c_size_t), intent(in), optional :: offset, bytes
        integer(c_size_t) :: actual_offset, actual_bytes

        actual_offset = 0_c_size_t
        if (present(offset)) actual_offset = offset
        actual_bytes = ggml_tensor_nbytes(tensor)
        if (present(bytes)) actual_bytes = bytes
        call c_ggml_backend_tensor_get(tensor, data, actual_offset, actual_bytes)
    end subroutine ggml_backend_get_tensor

    subroutine ggml_backend_tensor_copy(source, destination)
        type(c_ptr), intent(in) :: source, destination

        if (c_associated(source) .and. c_associated(destination)) then
            call c_ggml_backend_tensor_copy(source, destination)
        end if
    end subroutine ggml_backend_tensor_copy

    subroutine ggml_backend_synchronize(backend)
        type(c_ptr), intent(in) :: backend

        call c_ggml_backend_synchronize(backend)
    end subroutine ggml_backend_synchronize

    subroutine ggml_backend_free(backend)
        type(c_ptr), intent(inout) :: backend

        if (c_associated(backend)) call c_ggml_backend_free(backend)
        backend = c_null_ptr
    end subroutine ggml_backend_free

    subroutine ggml_backend_memory(device, free_bytes, total_bytes)
        type(c_ptr), intent(in) :: device
        integer(c_size_t), intent(out) :: free_bytes, total_bytes

        free_bytes = 0_c_size_t
        total_bytes = 0_c_size_t
        if (c_associated(device)) call c_ggml_backend_dev_memory(device, free_bytes, total_bytes)
    end subroutine ggml_backend_memory

    integer(c_size_t) function ggml_tensor_nbytes(tensor)
        type(c_ptr), intent(in) :: tensor

        ggml_tensor_nbytes = c_ggml_nbytes(tensor)
    end function ggml_tensor_nbytes

    integer(c_int64_t) function ggml_tensor_nelements(tensor)
        type(c_ptr), intent(in) :: tensor

        ggml_tensor_nelements = c_ggml_nelements(tensor)
    end function ggml_tensor_nelements

    integer(c_size_t) function ggml_tensor_type_size(value_type)
        integer(c_int), intent(in) :: value_type

        ggml_tensor_type_size = c_ggml_type_size(value_type)
    end function ggml_tensor_type_size

    integer(c_int64_t) function ggml_tensor_block_size(value_type)
        integer(c_int), intent(in) :: value_type

        ggml_tensor_block_size = c_ggml_blck_size(value_type)
    end function ggml_tensor_block_size

    integer(c_size_t) function ggml_tensor_element_size(tensor)
        type(c_ptr), intent(in) :: tensor

        ggml_tensor_element_size = c_ggml_element_size(tensor)
    end function ggml_tensor_element_size

    function ggml_tensor_data(tensor) result(data)
        type(c_ptr), intent(in) :: tensor
        type(c_ptr) :: data

        data = c_ggml_get_data(tensor)
    end function ggml_tensor_data

    integer(c_size_t) function ggml_tensor_overhead()
        ggml_tensor_overhead = c_ggml_tensor_overhead()
    end function ggml_tensor_overhead

    integer(c_int64_t) function ggml_time_us()
        ggml_time_us = c_ggml_time_us()
    end function ggml_time_us

    integer(c_size_t) function ggml_graph_overhead()
        ggml_graph_overhead = c_ggml_graph_overhead()
    end function ggml_graph_overhead

end module fortai_ggml
