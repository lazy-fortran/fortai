program fortai_bench_cpu_matvec
    use, intrinsic :: iso_fortran_env, only: int32, int64, real64
    use fortai_backend_cpu, only: cpu_matvec_inplace
    use fortai_status, only: status_t
    implicit none

    integer(int32) :: cols, i, rows, repeats
    integer(int64) :: clock_end, clock_start, clock_rate
    real(real64) :: elapsed, checksum
    real(real64), allocatable :: matrix(:,:), result(:), vector(:)
    type(status_t) :: stat

    rows = command_integer(1, 1024_int32)
    cols = command_integer(2, 1024_int32)
    repeats = command_integer(3, 20_int32)
    if (rows <= 0 .or. cols <= 0 .or. repeats <= 0) then
        error stop 'rows, columns, and repeats must be positive'
    end if

    allocate (matrix(rows, cols), vector(cols), result(rows))
    do i = 1, cols
        vector(i) = 0.25_real64 + real(mod(i, 17), real64) / 17.0_real64
    end do
    call fill_matrix(matrix)
    call cpu_matvec_inplace(matrix, vector, result, stat)
    if (.not. stat%is_ok()) error stop 'CPU reference matvec failed'

    call system_clock(clock_start, count_rate=clock_rate)
    do i = 1, repeats
        call cpu_matvec_inplace(matrix, vector, result, stat)
        if (.not. stat%is_ok()) error stop 'CPU benchmark matvec failed'
    end do
    call system_clock(clock_end)
    elapsed = real(clock_end - clock_start, real64) / real(clock_rate, real64)
    checksum = sum(result)

    print '(a)', 'backend,operation,rows,cols,repeats,seconds,matvec_per_second,checksum'
    print '(a,",",a,",",i0,",",i0,",",i0,",",es16.8,",",es16.8,",",es16.8)', &
        'cpu-native', 'matvec', rows, cols, repeats, elapsed, &
        real(repeats, real64) / elapsed, checksum

contains

    integer(int32) function command_integer(position, default_value)
        integer, intent(in) :: position
        integer(int32), intent(in) :: default_value
        character(len=64) :: argument
        integer :: io_status

        command_integer = default_value
        argument = ''
        call get_command_argument(position, argument, status=io_status)
        if (io_status == 0) then
            if (len_trim(argument) > 0) then
                read (argument, *, iostat=io_status) command_integer
                if (io_status /= 0) command_integer = default_value
            end if
        end if
    end function command_integer

    subroutine fill_matrix(values)
        real(real64), intent(out) :: values(:,:)
        integer :: i, j

        do j = 1, size(values, 2)
            do i = 1, size(values, 1)
                values(i, j) = 0.001_real64 * real(mod(i + 3 * j, 101), real64)
            end do
        end do
    end subroutine fill_matrix

end program fortai_bench_cpu_matvec
