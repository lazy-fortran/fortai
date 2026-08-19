module fortai_cache
    use, intrinsic :: iso_fortran_env, only: int32
    implicit none
    private

    type, public :: pack_key_t
        character(len=:), allocatable :: model
        character(len=:), allocatable :: backend
        character(len=:), allocatable :: variant
        integer(int32) :: abi_version = 1_int32
    contains
        procedure :: is_valid => pack_key_is_valid
    end type pack_key_t

    public :: cache_path

contains

    function cache_path(root, key) result(path)
        character(len=*), intent(in) :: root
        type(pack_key_t), intent(in) :: key
        character(len=:), allocatable :: path

        if (.not. key%is_valid()) then
            path = trim(root) // '/invalid.pack'
            return
        end if
        path = trim(root) // '/' // trim(key%model) // '/' // &
            trim(key%backend) // '-' // trim(key%variant) // '-abi' // &
            int_to_string(key%abi_version) // '.pack'
    end function cache_path

    pure logical function pack_key_is_valid(self)
        class(pack_key_t), intent(in) :: self

        pack_key_is_valid = .false.
        if (.not. allocated(self%model)) return
        if (.not. allocated(self%backend)) return
        if (.not. allocated(self%variant)) return
        if (len_trim(self%model) == 0) return
        if (len_trim(self%backend) == 0) return
        if (len_trim(self%variant) == 0) return
        if (self%abi_version <= 0_int32) return
        pack_key_is_valid = .true.
    end function pack_key_is_valid

    function int_to_string(value) result(text)
        integer(int32), intent(in) :: value
        character(len=:), allocatable :: text
        character(len=32) :: buffer

        write (buffer, '(i0)') value
        text = trim(buffer)
    end function int_to_string

end module fortai_cache
