program fortai_cuda_run
    use, intrinsic :: iso_c_binding, only: c_size_t
    use, intrinsic :: iso_fortran_env, only: int64, real32
    use fortai_backend_cuda, only: cuda_memory_info
    use fortai_qwen35_cpu, only: qwen35_cpu_model_t
    use fortai_status, only: status_t
    implicit none

    type(qwen35_cpu_model_t) :: model
    type(status_t) :: stat
    real(real32), allocatable :: logits(:), ranked_logits(:)
    character(len=512) :: model_path, argument
    integer(int64) :: token, steps, context, position, next_token
    integer(int64) :: first_position, last_position
    integer :: clock_start, clock_end, clock_rate, max_index, ios, device, trace_length
    integer :: disable_cuda_length
    integer :: sample_start, sample_end, top_count, trace_top_length, trace_top_ios, rank, top_index
    integer :: spec_count, emitted
    integer :: load_start, load_end, forward_start, forward_end
    real(real32) :: elapsed, load_seconds, forward_seconds, sample_seconds, tokens_per_second, checksum
    real(real32) :: greedy_sum
    character(len=16) :: trace_tokens, trace_top_tokens
    logical :: trace_enabled, disable_cuda, exclude_prompt
    character(len=8) :: disable_cuda_env
    character(len=16) :: exclude_prompt_text
    integer :: exclude_prompt_length
    integer(int64) :: speculative_tokens(32)
    integer(c_size_t) :: vram_free_before, vram_total_before
    integer(c_size_t) :: vram_free_after, vram_total_after
    logical :: vram_before_ok, vram_after_ok
    real(real32) :: maximum_logit

    call get_command_argument(1, model_path)
    if (len_trim(model_path) == 0) then
        print '(a)', 'usage: fortai_cuda_run MODEL.gguf [token_id] [steps] [context] [device]'
        error stop 2
    end if
    token = 0_int64
    steps = 2_int64
    context = 128_int64
    device = 0
    call get_command_argument(2, argument)
    if (len_trim(argument) > 0) read (argument, *, iostat=ios) token
    call get_command_argument(3, argument)
    if (len_trim(argument) > 0) read (argument, *, iostat=ios) steps
    call get_command_argument(4, argument)
    if (len_trim(argument) > 0) read (argument, *, iostat=ios) context
    call get_command_argument(5, argument)
    if (len_trim(argument) > 0) read (argument, *, iostat=ios) device
    if (steps <= 0_int64 .or. context <= 0_int64 .or. device < 0) error stop 2
    call cuda_memory_info(device, vram_free_before, vram_total_before, stat)
    vram_before_ok = stat%is_ok()
    call get_environment_variable('FORTAI_TRACE_TOKENS', trace_tokens, length=trace_length)
    trace_enabled = .false.
    if (trace_length > 0) trace_enabled = trace_tokens(1:trace_length) == '1'
    top_count = 0
    call get_environment_variable('FORTAI_TRACE_TOP_LOGITS', trace_top_tokens, &
        length=trace_top_length)
    if (trace_top_length > 0) then
        read (trace_top_tokens(1:trace_top_length), *, iostat=trace_top_ios) top_count
        if (trace_top_ios /= 0 .or. top_count < 2) error stop 2
    end if

    call system_clock(clock_start, clock_rate)
    load_start = clock_start
    call model%open(trim(model_path), context, stat)
    if (.not. stat%is_ok()) then
        print '(a)', 'FortAI model open failed: ' // stat%message
        error stop 1
    end if
    disable_cuda = .false.
    call get_environment_variable('FORTAI_DISABLE_CUDA', disable_cuda_env, length=disable_cuda_length)
    if (disable_cuda_length > 0) disable_cuda = disable_cuda_env(1:disable_cuda_length) == '1'
    if (.not. disable_cuda) then
        call model%enable_cuda(device, stat)
        if (.not. stat%is_ok()) then
            print '(a)', 'FortAI CUDA enable failed: ' // stat%message
            error stop 1
        end if
    end if
    call cuda_memory_info(device, vram_free_after, vram_total_after, stat)
    vram_after_ok = stat%is_ok()
    call system_clock(load_end)
    allocate (logits(model%vocabulary_size))
    if (top_count > 0) allocate (ranked_logits(model%vocabulary_size))
    call get_environment_variable('FORTAI_EXCLUDE_PROMPT', exclude_prompt_text, &
        length=exclude_prompt_length)
    exclude_prompt = exclude_prompt_length > 0 .and. &
        exclude_prompt_text(1:exclude_prompt_length) == '1'
    first_position = 0_int64
    last_position = steps - 1_int64
    if (exclude_prompt) then
        if (model%fast_enabled .and. top_count == 0) then
            call model%forward_greedy(token, 0_int64, next_token, greedy_sum, stat)
            if (.not. stat%is_ok()) then
                print '(a)', 'FortAI prompt forward failed: ' // stat%message
                error stop 1
            end if
            token = next_token
        else
            call model%forward(token, 0_int64, logits, stat)
            if (.not. stat%is_ok()) then
                print '(a)', 'FortAI prompt forward failed: ' // stat%message
                error stop 1
            end if
            max_index = maxloc(logits, dim=1)
            token = int(max_index - 1, int64)
        end if
        if (trace_enabled) print '(a,i0,a,i0)', 'token[', 0, ']=', token
        if (top_count > 0) then
            ranked_logits = logits
            maximum_logit = maxval(logits)
            do rank = 1, top_count
                top_index = maxloc(ranked_logits, dim=1)
                print '(a,i0,a,i0,a,i0,a,es16.8)', 'top_logit[', 0, ',', rank, &
                    ']=', top_index - 1, ',', ranked_logits(top_index) - maximum_logit
                ranked_logits(top_index) = -huge(0.0_real32)
            end do
        end if
        first_position = 1_int64
        last_position = steps - 1_int64
    end if
    checksum = 0.0_real32
    sample_seconds = 0.0_real32
    call system_clock(forward_start)
    position = first_position
    do while (position <= last_position)
        if (model%fast_enabled .and. top_count == 0) then
            call model%forward_greedy_speculative(token, position, speculative_tokens, &
                spec_count, greedy_sum, stat)
            if (.not. stat%is_ok()) then
                print '(a)', 'FortAI CUDA forward failed: ' // stat%message
                error stop 1
            end if
            checksum = checksum + greedy_sum
            do emitted = 1, min(spec_count, int(last_position - position + 1_int64))
                token = speculative_tokens(emitted)
                if (trace_enabled) print '(a,i0,a,i0)', 'token[', position + emitted - 1, ']=', token
            end do
            position = position + int(spec_count, int64)
        else
            call model%forward(token, position, logits, stat)
            if (.not. stat%is_ok()) then
                print '(a)', 'FortAI CUDA forward failed: ' // stat%message
                error stop 1
            end if
            checksum = checksum + sum(logits)
            if (top_count > 0) then
                ranked_logits = logits
                maximum_logit = maxval(logits)
                do rank = 1, top_count
                    top_index = maxloc(ranked_logits, dim=1)
                    print '(a,i0,a,i0,a,i0,a,es16.8)', 'top_logit[', position, ',', rank, &
                        ']=', top_index - 1, ',', ranked_logits(top_index) - maximum_logit
                    ranked_logits(top_index) = -huge(0.0_real32)
                end do
            end if
            call system_clock(sample_start)
            max_index = maxloc(logits, dim=1)
            call system_clock(sample_end)
            sample_seconds = sample_seconds + real(sample_end - sample_start, real32) / real(clock_rate, real32)
            next_token = int(max_index - 1, int64)
            token = next_token
            if (trace_enabled) print '(a,i0,a,i0)', 'token[', position, ']=', token
            position = position + 1_int64
        end if
    end do
    call system_clock(clock_end)
    forward_end = clock_end
    load_seconds = real(load_end - load_start, real32) / real(clock_rate, real32)
    forward_seconds = real(forward_end - forward_start, real32) / real(clock_rate, real32)
    elapsed = real(clock_end - clock_start, real32) / real(clock_rate, real32)
    tokens_per_second = real(steps, real32) / max(forward_seconds, 1.0e-6_real32)
    if (model%fast_enabled) then
        print '(a)', 'backend=fortai-llama-cpp-fastpath'
    else if (allocated(model%cuda_q4_weights)) then
        print '(a)', 'backend=fortai-cuda-host-q4-xl-ggml'
    else if (model%cuda_device_pipeline) then
        print '(a)', 'backend=fortai-cuda-device-recurrent-attention-q8'
    else
        print '(a)', 'backend=fortai-cuda-host-q8'
    end if
    print '(a,l1)', 'device_pipeline=', model%cuda_device_pipeline
    print '(a,l1)', 'cuda_graph_enabled=', model%cuda_graph_enabled
    print '(a,i0)', 'device=', device
    print '(a,i0)', 'vocabulary=', model%vocabulary_size
    print '(a,i0)', 'layers=', model%layer_count
    print '(a,i0)', 'steps=', steps
    print '(a,i0)', 'last_token=', token
    print '(a,es16.8)', 'logit_checksum=', checksum
    print '(a,es16.8)', 'load_seconds=', load_seconds
    print '(a,es16.8)', 'forward_seconds=', forward_seconds
    print '(a,es16.8)', 'sample_seconds=', sample_seconds
    print '(a,es16.8)', 'elapsed_seconds=', elapsed
    print '(a,es16.8)', 'tokens_per_second=', tokens_per_second
    if (vram_before_ok) then
        print '(a,i0)', 'vram_total_before_bytes=', vram_total_before
        print '(a,i0)', 'vram_free_before_bytes=', vram_free_before
    end if
    if (vram_after_ok) then
        print '(a,i0)', 'vram_total_after_bytes=', vram_total_after
        print '(a,i0)', 'vram_free_after_bytes=', vram_free_after
        if (vram_before_ok .and. vram_free_before >= vram_free_after) then
            print '(a,i0)', 'vram_delta_bytes=', vram_free_before - vram_free_after
        end if
    end if
    call model%close()
end program fortai_cuda_run
