module fortai_gguf
    use, intrinsic :: iso_fortran_env, only: int32, int64
    use fortai_status, only: FORTAI_INVALID, FORTAI_IO_ERROR, status_t
    implicit none
    private

    type, public :: gguf_header_t
        integer(int32) :: version = 0_int32
        integer(int64) :: tensor_count = 0_int64
        integer(int64) :: metadata_count = 0_int64
    end type gguf_header_t

    public :: gguf_read_header
    public :: gguf_validate_header

contains

    subroutine gguf_validate_header(magic, version, stat)
        character(len=*), intent(in) :: magic
        integer(int32), intent(in) :: version
        type(status_t), intent(out) :: stat

        call stat%clear()
        if (len(magic) < 4) then
            call stat%set(FORTAI_INVALID, 'GGUF magic must contain four bytes')
            return
        end if
        if (magic(1:4) /= 'GGUF') then
            call stat%set(FORTAI_INVALID, 'file does not start with GGUF magic')
            return
        end if
        if (version < 1_int32 .or. version > 3_int32) then
            call stat%set(FORTAI_INVALID, 'unsupported GGUF version')
        end if
    end subroutine gguf_validate_header

    subroutine gguf_read_header(path, header, stat)
        character(len=*), intent(in) :: path
        type(gguf_header_t), intent(out) :: header
        type(status_t), intent(out) :: stat
        character(len=4) :: magic
        integer(int32) :: version
        integer(int64) :: tensor_count, metadata_count
        integer :: io_status, unit

        header = gguf_header_t()
        call stat%clear()
        open (newunit=unit, file=path, access='stream', form='unformatted', &
            status='old', action='read', iostat=io_status)
        if (io_status /= 0) then
            call stat%set(FORTAI_IO_ERROR, 'could not open GGUF file')
            return
        end if

        read (unit, iostat=io_status) magic
        if (io_status == 0) read (unit, iostat=io_status) version
        if (io_status == 0) read (unit, iostat=io_status) tensor_count
        if (io_status == 0) read (unit, iostat=io_status) metadata_count
        close (unit)
        if (io_status /= 0) then
            call stat%set(FORTAI_IO_ERROR, 'could not read GGUF header')
            return
        end if

        call gguf_validate_header(magic, version, stat)
        if (.not. stat%is_ok()) return
        if (tensor_count < 0_int64 .or. metadata_count < 0_int64) then
            call stat%set(FORTAI_INVALID, 'GGUF counts cannot be negative')
            return
        end if
        header%version = version
        header%tensor_count = tensor_count
        header%metadata_count = metadata_count
    end subroutine gguf_read_header

end module fortai_gguf
