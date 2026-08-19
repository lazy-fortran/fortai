module fortai
    use, intrinsic :: iso_fortran_env, only: int32
    use fortai_device, only: device_cpu, device_t
    use fortai_backend_cuda, only: cuda_q8_context_t, cuda_q8_matvec_host, &
        cuda_q8_matvec_resident, cuda_q8_weights_t
    use fortai_status, only: FORTAI_OK, status_t
    use fortai_tensor, only: tensor_t
    implicit none
    private

    public :: FORTAI_OK
    public :: device_cpu
    public :: device_t
    public :: cuda_q8_context_t
    public :: cuda_q8_matvec_host
    public :: cuda_q8_matvec_resident
    public :: cuda_q8_weights_t
    public :: fortai_version
    public :: status_t
    public :: tensor_t

contains

    pure function fortai_version() result(version)
        character(len=5) :: version

        version = '0.1.0'
    end function fortai_version

    pure integer(int32) function fortai_api_level()
        fortai_api_level = 1_int32
    end function fortai_api_level

end module fortai
