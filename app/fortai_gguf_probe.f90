program fortai_gguf_probe
    use, intrinsic :: iso_fortran_env, only: int64, real32
    use fortai_gguf_runtime, only: gguf_file_t
    use fortai_status, only: status_t
    implicit none

    type(gguf_file_t) :: file
    type(status_t) :: stat
    real(real32), allocatable :: values(:)
    character(len=512) :: path
    integer :: index, i

    call get_command_argument(1, path)
    if (len_trim(path) == 0) error stop 2
    call file % open(trim(path), stat)
    if (.not. stat % is_ok()) then
        print '(a)', stat % message
        error stop 1
    end if
    index = file % find_tensor('token_embd.weight')
    if (index == 0) error stop 1
    print '(a,i0)', 'type=', file % tensors(index) % value_type
    print '(a,2(i0,1x))', 'shape=', file % tensors(index) % shape
    allocate (values(file % tensors(index) % shape(1)))
    call file % tensors(index) % get_row(1_int64, values, stat)
    if (.not. stat % is_ok()) then
        print '(a)', stat % message
        error stop 1
    end if
    do i = 1, min(8, size(values))
        print '(a,i0,a,es16.8)', 'value[', i, ']=', values(i)
    end do
end program fortai_gguf_probe
