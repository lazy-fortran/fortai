program fortran_cuda_backend_smoke
    use, intrinsic :: iso_c_binding, only: c_float, c_int8_t, c_null_ptr, c_ptr, c_size_t
    use fortai_backend_cuda, only: cuda_q8_context_t, cuda_q8_matvec_host, &
        cuda_q8_matvec_resident, cuda_q8_weights_t
    use fortai_status, only: status_t
    implicit none

    integer, parameter :: rows = 128, width = 128, blocks = width / 32
    integer, parameter :: block_bytes = 34
    integer(c_int8_t), allocatable, target :: weights_host(:), activation_host(:)
    real(c_float), allocatable, target :: host_output(:), device_output(:)
    type(cuda_q8_context_t) :: context
    type(cuda_q8_weights_t) :: weights
    type(c_ptr) :: device_activation = c_null_ptr, device_result = c_null_ptr
    type(status_t) :: stat
    real(c_float) :: host_ms, kernel_ms
    integer :: block, index, offset

    allocate (weights_host(rows * blocks * block_bytes))
    allocate (activation_host(blocks * block_bytes))
    allocate (host_output(rows), device_output(rows))
    weights_host = 0_c_int8_t
    activation_host = 0_c_int8_t
    do block = 0, blocks - 1
        offset = block * block_bytes
        activation_host(offset + 1) = 0_c_int8_t
        activation_host(offset + 2) = 60_c_int8_t
        do index = 0, 31
            activation_host(offset + 3 + index) = int(mod(index, 15) - 7, c_int8_t)
        end do
    end do
    do index = 0, rows * blocks - 1
        offset = index * block_bytes
        weights_host(offset + 1) = 0_c_int8_t
        weights_host(offset + 2) = 60_c_int8_t
        do block = 0, 31
            weights_host(offset + 3 + block) = int(mod(index + block, 17) - 8, c_int8_t)
        end do
    end do

    call context%create(0, stat)
    call require_ok(stat, 'context create')
    call weights%upload(context, weights_host, int(size(weights_host), c_size_t), rows, width, stat)
    call require_ok(stat, 'weight upload')
    call cuda_q8_matvec_host(context, weights, activation_host, &
        int(size(activation_host), c_size_t), host_output, &
        int(size(host_output) * 4, c_size_t), host_ms, stat)
    call require_ok(stat, 'host matvec')

    call context%allocate_buffer(int(size(activation_host), c_size_t), device_activation, stat)
    call require_ok(stat, 'activation allocation')
    call context%allocate_buffer(int(size(device_output) * 4, c_size_t), device_result, stat)
    call require_ok(stat, 'output allocation')
    call context%upload(device_activation, activation_host, &
        int(size(activation_host), c_size_t), stat)
    call require_ok(stat, 'activation upload')
    call cuda_q8_matvec_resident(context, weights, device_activation, device_result, kernel_ms, stat)
    call require_ok(stat, 'resident matvec')
    call context%download_real(device_result, device_output, stat)
    call require_ok(stat, 'output download')

    if (any(host_output /= host_output) .or. any(device_output /= device_output)) &
        error stop 'CUDA Fortran binding returned NaN'
    print '(a, f10.5, a, f10.5, a)', &
        '{"implementation":"fortai-fortran-cuda-q8-abi","device":0,"host_elapsed_ms":', &
        host_ms, ',"resident_kernel_ms":', kernel_ms, &
        ',"finite":true}'

    call context%free_buffer(device_result, stat)
    call require_ok(stat, 'output free')
    call context%free_buffer(device_activation, stat)
    call require_ok(stat, 'activation free')
    call weights%destroy(stat)
    call require_ok(stat, 'weight free')
    call context%destroy(stat)
    call require_ok(stat, 'context destroy')

contains

    subroutine require_ok(value, operation)
        type(status_t), intent(in) :: value
        character(len=*), intent(in) :: operation

        if (.not. value%is_ok()) error stop operation
    end subroutine require_ok

end program fortran_cuda_backend_smoke
