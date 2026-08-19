module fortai_arena
    use, intrinsic :: iso_fortran_env, only: int64
    use fortai_status, only: FORTAI_INVALID, FORTAI_OUT_OF_MEMORY, status_t
    implicit none
    private

    type, public :: arena_t
        integer(int64) :: capacity_bytes = 0_int64
        integer(int64) :: used_bytes = 0_int64
    contains
        procedure :: available => arena_available
        procedure :: init => arena_init
        procedure :: reset => arena_reset
        procedure :: reserve => arena_reserve
    end type arena_t

contains

    subroutine arena_init(self, capacity_bytes, stat)
        class(arena_t), intent(inout) :: self
        integer(int64), intent(in) :: capacity_bytes
        type(status_t), intent(out), optional :: stat

        if (present(stat)) call stat%clear()
        self%capacity_bytes = 0_int64
        self%used_bytes = 0_int64
        if (capacity_bytes < 0_int64) then
            if (present(stat)) call stat%set(FORTAI_INVALID, &
                'arena capacity cannot be negative')
            return
        end if
        self%capacity_bytes = capacity_bytes
    end subroutine arena_init

    integer(int64) function arena_available(self)
        class(arena_t), intent(in) :: self

        arena_available = self%capacity_bytes - self%used_bytes
    end function arena_available

    subroutine arena_reset(self)
        class(arena_t), intent(inout) :: self

        self%used_bytes = 0_int64
    end subroutine arena_reset

    subroutine arena_reserve(self, bytes, stat)
        class(arena_t), intent(inout) :: self
        integer(int64), intent(in) :: bytes
        type(status_t), intent(out), optional :: stat

        if (present(stat)) call stat%clear()
        if (bytes < 0_int64) then
            if (present(stat)) call stat%set(FORTAI_INVALID, &
                'arena reservation cannot be negative')
            return
        end if
        if (bytes > self%capacity_bytes - self%used_bytes) then
            if (present(stat)) call stat%set(FORTAI_OUT_OF_MEMORY, &
                'arena capacity exceeded')
            return
        end if
        self%used_bytes = self%used_bytes + bytes
    end subroutine arena_reserve

end module fortai_arena
