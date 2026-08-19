module fortai_backend_cuda
    use, intrinsic :: iso_c_binding, only: c_float, c_int, c_int8_t, c_loc, c_ptr, &
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
        procedure :: allocate_buffer => cuda_q8_allocate_buffer
        procedure :: free_buffer => cuda_q8_free_buffer
        procedure :: upload => cuda_q8_upload
        procedure :: download => cuda_q8_download
        procedure :: download_real => cuda_q8_download_real
        procedure :: last_error => cuda_q8_last_error
    end type cuda_q8_context_t

    type, public :: cuda_q8_weights_t
        type(c_ptr) :: handle = c_null_ptr
    contains
        procedure :: upload => cuda_q8_weights_upload
        procedure :: destroy => cuda_q8_weights_destroy
    end type cuda_q8_weights_t

    type, public :: cuda_qwen35_recurrent_t
        type(c_ptr) :: handle = c_null_ptr
    contains
        procedure :: create => cuda_qwen35_recurrent_create
        procedure :: destroy => cuda_qwen35_recurrent_destroy
        procedure :: reset => cuda_qwen35_recurrent_reset
        procedure :: run => cuda_qwen35_recurrent_run
    end type cuda_qwen35_recurrent_t

    public :: cuda_q8_matvec_host
    public :: cuda_q8_matvec_host_pair
    public :: cuda_q8_matvec_host_triplet
    public :: cuda_q8_matvec_host_triplet_contiguous
    public :: cuda_q8_ffn_host
    public :: cuda_q8_matvec_resident

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

        function c_last_error(context) bind(C, name='fortai_cuda_q8_last_error') result(message)
            import c_ptr
            type(c_ptr), value :: context
            type(c_ptr) :: message
        end function c_last_error
    end interface

contains

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
