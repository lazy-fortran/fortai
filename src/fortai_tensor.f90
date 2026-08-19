module fortai_tensor
    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    use fortai_status, only: FORTAI_INVALID, status_t
    implicit none
    private

    type, public :: tensor_t
        real(real64), allocatable :: data(:)
        integer(int32), allocatable :: shape(:)
        logical :: resident = .false.
    contains
        procedure :: clear => tensor_clear
        procedure :: fill => tensor_fill
        procedure :: init => tensor_init
        procedure :: numel => tensor_numel
        procedure :: value_at => tensor_value_at
    end type tensor_t

contains

    subroutine tensor_clear(self)
        class(tensor_t), intent(inout) :: self

        if (allocated(self%data)) deallocate (self%data)
        if (allocated(self%shape)) deallocate (self%shape)
        self%resident = .false.
    end subroutine tensor_clear

    subroutine tensor_fill(self, value)
        class(tensor_t), intent(inout) :: self
        real(real64), intent(in) :: value

        if (allocated(self%data)) self%data = value
    end subroutine tensor_fill

    subroutine tensor_init(self, extent, stat)
        class(tensor_t), intent(inout) :: self
        integer(int32), intent(in) :: extent(:)
        type(status_t), intent(out), optional :: stat
        integer(int32) :: i

        if (present(stat)) call stat%clear()
        call self%clear()
        if (size(extent) == 0) then
            if (present(stat)) call stat%set(FORTAI_INVALID, 'tensor rank must be positive')
            return
        end if
        do i = 1, size(extent)
            if (extent(i) <= 0) then
                if (present(stat)) call stat%set(FORTAI_INVALID, &
                    'tensor extents must be positive')
                return
            end if
        end do

        allocate (self%shape, source=extent)
        allocate (self%data(product(extent)))
        self%data = 0.0_real64
    end subroutine tensor_init

    integer(int64) function tensor_numel(self)
        class(tensor_t), intent(in) :: self

        if (allocated(self%data)) then
            tensor_numel = int(size(self%data), int64)
        else
            tensor_numel = 0_int64
        end if
    end function tensor_numel

    real(real64) function tensor_value_at(self, index, stat)
        class(tensor_t), intent(in) :: self
        integer(int32), intent(in) :: index
        type(status_t), intent(out), optional :: stat

        if (present(stat)) call stat%clear()
        tensor_value_at = 0.0_real64
        if (.not. allocated(self%data)) then
            if (present(stat)) call stat%set(FORTAI_INVALID, 'tensor is not allocated')
            return
        end if
        if (index < 1) then
            if (present(stat)) call stat%set(FORTAI_INVALID, 'tensor index is below one')
            return
        end if
        if (index > size(self%data)) then
            if (present(stat)) call stat%set(FORTAI_INVALID, 'tensor index is out of range')
            return
        end if
        tensor_value_at = self%data(index)
    end function tensor_value_at

end module fortai_tensor
