module fortai_string
    !! A compact growable string used by the resident service and protocol code.
    !!
    !! The allocation is geometric, so appending a request or response is
    !! amortized linear and does not create a temporary character array for
    !! every fragment.  The C-boundary conversion routines are deliberately
    !! confined to this module.
    use, intrinsic :: iso_c_binding, only: c_char, c_null_char
    implicit none
    private

    type, public :: string_t
        private
        character(len=:), allocatable :: value
        integer :: used = 0
    contains
        procedure, public :: clear => string_clear
        procedure, public :: reserve => string_reserve
        procedure, public :: append => string_append
        procedure, public :: append_string => string_append_string
        procedure, public :: append_char => string_append_char
        procedure, public :: append_int => string_append_int
        procedure, public :: append_logical => string_append_logical
        procedure, public :: set => string_set
        procedure, public :: equals => string_equals
        procedure, public :: length => string_length
        procedure, public :: as_character => string_as_character
        procedure, public :: from_c => string_from_c
        procedure, public :: to_c => string_to_c
    end type string_t

contains

    subroutine string_clear(self)
        class(string_t), intent(inout) :: self
        self%used = 0
    end subroutine string_clear

    subroutine string_reserve(self, required)
        class(string_t), intent(inout) :: self
        integer, intent(in) :: required
        character(len=:), allocatable :: grown
        integer :: capacity

        if (required <= 0) return
        if (allocated(self%value) .and. len(self%value) >= required) return
        if (allocated(self%value)) then
            capacity = max(16, len(self%value))
        else
            capacity = 16
        end if
        do while (capacity < required)
            capacity = max(capacity + 1, 2 * capacity)
        end do
        allocate(character(len=capacity) :: grown)
        grown(:) = ' '
        if (self%used > 0 .and. allocated(self%value)) grown(:self%used) = self%value(:self%used)
        call move_alloc(grown, self%value)
    end subroutine string_reserve

    subroutine string_append(self, fragment)
        class(string_t), intent(inout) :: self
        character(len=*), intent(in) :: fragment
        integer :: n

        n = len(fragment)
        if (n == 0) return
        call self%reserve(self%used + n)
        self%value(self%used + 1:self%used + n) = fragment
        self%used = self%used + n
    end subroutine string_append

    subroutine string_append_char(self, fragment)
        class(string_t), intent(inout) :: self
        character(len=1), intent(in) :: fragment
        call self%append(fragment)
    end subroutine string_append_char

    subroutine string_append_string(self, fragment)
        class(string_t), intent(inout) :: self
        class(string_t), intent(in) :: fragment
        if (fragment%used == 0) return
        call self%reserve(self%used + fragment%used)
        self%value(self%used + 1:self%used + fragment%used) = fragment%value(:fragment%used)
        self%used = self%used + fragment%used
    end subroutine string_append_string

    subroutine string_append_int(self, number)
        class(string_t), intent(inout) :: self
        integer, intent(in) :: number
        character(len=32) :: buffer
        write(buffer, '(i0)') number
        call self%append(buffer(:len_trim(buffer)))
    end subroutine string_append_int

    subroutine string_append_logical(self, value)
        class(string_t), intent(inout) :: self
        logical, intent(in) :: value
        if (value) then
            call self%append('true')
        else
            call self%append('false')
        end if
    end subroutine string_append_logical

    subroutine string_set(self, text)
        class(string_t), intent(inout) :: self
        character(len=*), intent(in) :: text
        call self%clear()
        call self%append(text)
    end subroutine string_set

    logical function string_equals(self, text)
        class(string_t), intent(in) :: self
        character(len=*), intent(in) :: text

        string_equals = .false.
        if (self%used /= len(text)) return
        if (self%used == 0) then
            string_equals = .true.
        else if (allocated(self%value)) then
            string_equals = self%value(:self%used) == text
        end if
    end function string_equals

    integer function string_length(self)
        class(string_t), intent(in) :: self
        string_length = self%used
    end function string_length

    function string_as_character(self) result(text)
        class(string_t), intent(in) :: self
        character(len=:), allocatable :: text
        allocate(character(len=self%used) :: text)
        if (self%used > 0) text = self%value(:self%used)
    end function string_as_character

    subroutine string_from_c(self, input, length)
        class(string_t), intent(inout) :: self
        character(kind=c_char), intent(in) :: input(*)
        integer, intent(in), optional :: length
        integer :: n, i

        if (present(length)) then
            n = max(0, length)
        else
            n = 0
            do while (input(n + 1) /= c_null_char)
                n = n + 1
            end do
        end if
        call self%clear()
        call self%reserve(n)
        do i = 1, n
            self%value(i:i) = achar(iachar(input(i)))
        end do
        self%used = n
    end subroutine string_from_c

    subroutine string_to_c(self, output, capacity, required)
        class(string_t), intent(in) :: self
        character(kind=c_char), intent(out) :: output(*)
        integer, intent(in) :: capacity
        integer, intent(out), optional :: required
        integer :: i

        if (present(required)) required = self%used
        if (capacity <= 0) return
        do i = 1, min(self%used, capacity - 1)
            output(i) = self%value(i:i)
        end do
        output(min(self%used, capacity - 1) + 1) = c_null_char
    end subroutine string_to_c

end module fortai_string
