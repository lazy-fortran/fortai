module fortai_device
    use, intrinsic :: iso_fortran_env, only: int32
    implicit none
    private

    integer(int32), parameter, public :: FORTAI_BACKEND_CPU = 1_int32
    integer(int32), parameter, public :: FORTAI_BACKEND_CUDA = 2_int32
    integer(int32), parameter, public :: FORTAI_BACKEND_METAL = 3_int32
    integer(int32), parameter, public :: FORTAI_BACKEND_MLX = 4_int32
    integer(int32), parameter, public :: FORTAI_BACKEND_HIP = 5_int32
    integer(int32), parameter, public :: FORTAI_BACKEND_SYCL = 6_int32
    integer(int32), parameter, public :: FORTAI_BACKEND_VULKAN = 7_int32
    integer(int32), parameter, public :: FORTAI_BACKEND_TINYGPU_NV = 8_int32

    type, public :: device_t
        integer(int32) :: backend = FORTAI_BACKEND_CPU
        integer(int32) :: index = 0_int32
        character(len=:), allocatable :: name
    contains
        procedure :: is_supported => device_is_supported
        procedure :: label => device_label
    end type device_t

    public :: device_cpu
    public :: backend_name

contains

    function device_cpu(index, name) result(device)
        integer(int32), intent(in), optional :: index
        character(len=*), intent(in), optional :: name
        type(device_t) :: device

        device%backend = FORTAI_BACKEND_CPU
        if (present(index)) device%index = index
        if (present(name)) then
            device%name = name
        else
            device%name = 'host-cpu'
        end if
    end function device_cpu

    pure function backend_name(backend) result(name)
        integer(int32), intent(in) :: backend
        character(len=:), allocatable :: name

        select case (backend)
        case (FORTAI_BACKEND_CPU)
            name = 'cpu'
        case (FORTAI_BACKEND_CUDA)
            name = 'cuda'
        case (FORTAI_BACKEND_METAL)
            name = 'metal'
        case (FORTAI_BACKEND_MLX)
            name = 'mlx'
        case (FORTAI_BACKEND_HIP)
            name = 'hip'
        case (FORTAI_BACKEND_SYCL)
            name = 'sycl'
        case (FORTAI_BACKEND_VULKAN)
            name = 'vulkan'
        case (FORTAI_BACKEND_TINYGPU_NV)
            name = 'tinygpu-nv'
        case default
            name = 'unknown'
        end select
    end function backend_name

    logical function device_is_supported(self)
        class(device_t), intent(in) :: self

        device_is_supported = self%backend == FORTAI_BACKEND_CPU
    end function device_is_supported

    function device_label(self) result(label)
        class(device_t), intent(in) :: self
        character(len=:), allocatable :: label

        if (allocated(self%name)) then
            label = self%name
        else
            label = backend_name(self%backend)
        end if
    end function device_label

end module fortai_device
