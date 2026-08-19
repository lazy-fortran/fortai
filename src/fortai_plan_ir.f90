module fortai_plan_ir
    use, intrinsic :: iso_fortran_env, only: int32
    use fortai_status, only: FORTAI_INVALID, status_t
    implicit none
    private

    type, public :: plan_step_t
        character(len=:), allocatable :: operation
        character(len=:), allocatable :: kernel
        integer(int32) :: batch_size = 1_int32
    end type plan_step_t

    type, public :: plan_ir_t
        type(plan_step_t), allocatable :: steps(:)
    contains
        procedure :: append => plan_append
        procedure :: clear => plan_clear
        procedure :: count => plan_count
        procedure :: init => plan_init
        procedure :: validate => plan_validate
    end type plan_ir_t

contains

    subroutine plan_append(self, operation, kernel, batch_size, stat)
        class(plan_ir_t), intent(inout) :: self
        character(len=*), intent(in) :: operation
        character(len=*), intent(in) :: kernel
        integer(int32), intent(in), optional :: batch_size
        type(status_t), intent(out), optional :: stat
        type(plan_step_t), allocatable :: grown(:)
        integer(int32) :: batch, old_count

        if (present(stat)) call stat%clear()
        batch = 1_int32
        if (present(batch_size)) batch = batch_size
        if (len_trim(operation) == 0) then
            if (present(stat)) call stat%set(FORTAI_INVALID, &
                'plan operation cannot be empty')
            return
        end if
        if (len_trim(kernel) == 0) then
            if (present(stat)) call stat%set(FORTAI_INVALID, &
                'plan kernel cannot be empty')
            return
        end if
        if (batch <= 0_int32) then
            if (present(stat)) call stat%set(FORTAI_INVALID, &
                'plan batch size must be positive')
            return
        end if

        old_count = self%count()
        allocate (grown(old_count + 1))
        if (old_count > 0) grown(1:old_count) = self%steps
        grown(old_count + 1)%operation = operation
        grown(old_count + 1)%kernel = kernel
        grown(old_count + 1)%batch_size = batch
        call move_alloc(grown, self%steps)
    end subroutine plan_append

    subroutine plan_clear(self)
        class(plan_ir_t), intent(inout) :: self

        if (allocated(self%steps)) deallocate (self%steps)
    end subroutine plan_clear

    integer(int32) function plan_count(self)
        class(plan_ir_t), intent(in) :: self

        if (allocated(self%steps)) then
            plan_count = int(size(self%steps), int32)
        else
            plan_count = 0_int32
        end if
    end function plan_count

    subroutine plan_init(self)
        class(plan_ir_t), intent(inout) :: self

        call self%clear()
    end subroutine plan_init

    subroutine plan_validate(self, stat)
        class(plan_ir_t), intent(in) :: self
        type(status_t), intent(out) :: stat
        integer(int32) :: i

        call stat%clear()
        if (.not. allocated(self%steps)) then
            call stat%set(FORTAI_INVALID, 'execution plan has no steps')
            return
        end if
        if (size(self%steps) == 0) then
            call stat%set(FORTAI_INVALID, 'execution plan has no steps')
            return
        end if
        do i = 1, size(self%steps)
            if (.not. allocated(self%steps(i)%operation)) then
                call stat%set(FORTAI_INVALID, 'plan operation is missing')
                return
            end if
            if (.not. allocated(self%steps(i)%kernel)) then
                call stat%set(FORTAI_INVALID, 'plan kernel is missing')
                return
            end if
            if (self%steps(i)%batch_size <= 0_int32) then
                call stat%set(FORTAI_INVALID, 'plan batch size is invalid')
                return
            end if
        end do
    end subroutine plan_validate

end module fortai_plan_ir
