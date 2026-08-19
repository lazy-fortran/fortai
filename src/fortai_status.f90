module fortai_status
    use, intrinsic :: iso_fortran_env, only: int32
    implicit none
    private

    integer(int32), parameter, public :: FORTAI_OK = 0_int32
    integer(int32), parameter, public :: FORTAI_INVALID = 1_int32
    integer(int32), parameter, public :: FORTAI_UNSUPPORTED = 2_int32
    integer(int32), parameter, public :: FORTAI_IO_ERROR = 3_int32
    integer(int32), parameter, public :: FORTAI_OUT_OF_MEMORY = 4_int32

    type, public :: status_t
        integer(int32) :: code = FORTAI_OK
        character(len=:), allocatable :: message
    contains
        procedure :: clear => status_clear
        procedure :: is_ok => status_is_ok
        procedure :: set => status_set
    end type status_t

contains

    subroutine status_clear(self)
        class(status_t), intent(inout) :: self

        self%code = FORTAI_OK
        if (allocated(self%message)) deallocate (self%message)
    end subroutine status_clear

    logical function status_is_ok(self)
        class(status_t), intent(in) :: self

        status_is_ok = self%code == FORTAI_OK
    end function status_is_ok

    subroutine status_set(self, code, message)
        class(status_t), intent(inout) :: self
        integer(int32), intent(in) :: code
        character(len=*), intent(in), optional :: message

        self%code = code
        if (allocated(self%message)) deallocate (self%message)
        if (present(message)) then
            self%message = message
        else
            self%message = ''
        end if
    end subroutine status_set

end module fortai_status
