program fortai_cpu_run
    use, intrinsic :: iso_fortran_env, only: int64, real32
    use fortai_qwen35_cpu, only: qwen35_cpu_model_t
    use fortai_status, only: status_t
    implicit none

    type(qwen35_cpu_model_t) :: model
    type(status_t) :: stat
    real(real32), allocatable :: logits(:)
    character(len=512) :: model_path, argument
    integer(int64) :: token, steps, context, position, next_token
    integer :: clock_start, clock_end, clock_rate, i, ios
    integer :: load_start, load_end, forward_start, forward_end
    real(real32) :: elapsed, load_seconds, forward_seconds, tokens_per_second, checksum

    call get_command_argument(1, model_path)
    if (len_trim(model_path) == 0) then
        print '(a)', 'usage: fortai_cpu_run MODEL.gguf [token_id] [steps] [context]'
        error stop 2
    end if
    token = 0_int64
    steps = 2_int64
    context = 128_int64
    call get_command_argument(2, argument)
    if (len_trim(argument) > 0) read (argument, *, iostat=ios) token
    call get_command_argument(3, argument)
    if (len_trim(argument) > 0) read (argument, *, iostat=ios) steps
    call get_command_argument(4, argument)
    if (len_trim(argument) > 0) read (argument, *, iostat=ios) context
    if (steps <= 0_int64 .or. context <= 0_int64) error stop 2

    call system_clock(clock_start, clock_rate)
    load_start = clock_start
    call model%open(trim(model_path), context, stat)
    if (.not. stat%is_ok()) then
        print '(a)', 'FortAI model open failed: ' // stat%message
        error stop 1
    end if
    call system_clock(load_end)
    allocate (logits(model%vocabulary_size))
    checksum = 0.0_real32
    call system_clock(forward_start)
    do position = 0_int64, steps - 1_int64
        call model%forward(token, position, logits, stat)
        if (.not. stat%is_ok()) then
            print '(a)', 'FortAI forward failed: ' // stat%message
            error stop 1
        end if
        checksum = checksum + sum(logits)
        next_token = 0_int64
        do i = 2, size(logits)
            if (logits(i) > logits(next_token + 1_int64)) next_token = int(i - 1, int64)
        end do
        token = next_token
    end do
    call system_clock(clock_end)
    forward_end = clock_end
    load_seconds = real(load_end - load_start, real32) / real(clock_rate, real32)
    forward_seconds = real(forward_end - forward_start, real32) / real(clock_rate, real32)
    elapsed = real(clock_end - clock_start, real32) / real(clock_rate, real32)
    tokens_per_second = real(steps, real32) / max(forward_seconds, 1.0e-6_real32)
    print '(a,i0)', 'vocabulary=', model%vocabulary_size
    print '(a,i0)', 'layers=', model%layer_count
    print '(a,i0)', 'steps=', steps
    print '(a,i0)', 'last_token=', token
    print '(a,es16.8)', 'logit_checksum=', checksum
    print '(a,es16.8)', 'load_seconds=', load_seconds
    print '(a,es16.8)', 'forward_seconds=', forward_seconds
    print '(a,es16.8)', 'elapsed_seconds=', elapsed
    print '(a,es16.8)', 'tokens_per_second=', tokens_per_second
end program fortai_cpu_run
