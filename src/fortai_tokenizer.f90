module fortai_tokenizer
    use, intrinsic :: iso_fortran_env, only: int32
    implicit none
    private

    integer(int32), parameter, public :: FORTAI_TOKEN_BOS = 1_int32
    integer(int32), parameter, public :: FORTAI_TOKEN_EOS = 2_int32

    type, public :: tokenizer_t
        integer(int32) :: vocab_size = 259_int32
    contains
        procedure :: decode => tokenizer_decode
        procedure :: encode => tokenizer_encode
    end type tokenizer_t

contains

    subroutine tokenizer_encode(self, text, ids)
        class(tokenizer_t), intent(in) :: self
        character(len=*), intent(in) :: text
        integer(int32), allocatable, intent(out) :: ids(:)
        integer(int32) :: i, length

        length = int(len_trim(text), int32)
        allocate (ids(length + 2))
        ids(1) = FORTAI_TOKEN_BOS
        do i = 1, length
            ids(i + 1) = 3_int32 + int(iachar(text(i:i)), int32)
        end do
        ids(length + 2) = FORTAI_TOKEN_EOS
    end subroutine tokenizer_encode

    function tokenizer_decode(self, ids) result(text)
        class(tokenizer_t), intent(in) :: self
        integer(int32), intent(in) :: ids(:)
        character(len=:), allocatable :: text
        integer(int32) :: i, length

        length = 0_int32
        do i = 1, size(ids)
            if (ids(i) >= 3_int32 .and. ids(i) <= 258_int32) length = length + 1_int32
        end do
        allocate (character(len=length) :: text)
        length = 0_int32
        do i = 1, size(ids)
            if (ids(i) >= 3_int32 .and. ids(i) <= 258_int32) then
                length = length + 1_int32
                text(length:length) = achar(ids(i) - 3_int32)
            end if
        end do
    end function tokenizer_decode

end module fortai_tokenizer
