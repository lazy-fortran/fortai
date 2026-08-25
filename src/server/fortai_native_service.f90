module fortai_native_service
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char
    use, intrinsic :: iso_fortran_env, only: error_unit, int32, int64, real32
    use fortai_native_tokenizer, only: fortai_native_tokenizer_t
    use fortai_qwen35_cpu, only: qwen35_cpu_model_t
    use fortai_string, only: string_t
    use fortai_status, only: status_t
    implicit none
    private

    type(qwen35_cpu_model_t), save :: service_model
    type(fortai_native_tokenizer_t), save :: service_tokenizer
    logical, save :: service_ready = .false.

    public :: fortai_native_service_init
    public :: fortai_native_service_complete
    public :: fortai_native_service_complete_text
    public :: fortai_native_service_complete_text_options
    public :: fortai_native_service_close

contains

    logical function finite_real32(value)
        real(real32), intent(in) :: value
        integer(int32) :: bits
        integer(int32), parameter :: exponent_mask = int(z'7f800000', int32)
        bits = transfer(value, bits)
        finite_real32 = iand(bits, exponent_mask) /= exponent_mask
    end function finite_real32

    function c_string(value) result(text)
        character(kind=c_char), intent(in) :: value(*)
        character(len=:), allocatable :: text
        integer :: length, i
        length = 0
        do while (value(length + 1) /= c_null_char)
            length = length + 1
            if (length > 16 * 1024 * 1024) exit
        end do
        allocate(character(len=length) :: text)
        do i = 1, length
            text(i:i) = achar(iachar(value(i)))
        end do
    end function c_string

    integer(c_int) function fortai_native_service_init(path, context_size, threads, gpu_layers, &
            main_gpu, require_cuda, vocab, layers) bind(C, name='fortai_native_service_init')
        character(kind=c_char), intent(in) :: path(*)
        integer(c_int), value, intent(in) :: context_size, threads, gpu_layers, main_gpu, require_cuda
        integer(c_int), intent(out) :: vocab, layers
        character(len=:), allocatable :: model_path
        type(status_t) :: stat
        logical :: tokenizer_ok

        fortai_native_service_init = 1_c_int
        vocab = 0_c_int
        layers = 0_c_int
        call fortai_native_service_close()
        model_path = c_string(path)
        if (len_trim(model_path) == 0 .or. context_size <= 0 .or. threads <= 0) return
        call service_model%open(trim(model_path), int(context_size, int64), stat)
        if (.not. stat%is_ok()) then
            write(error_unit, '(a)') 'fortai-native: model open failed: ' // trim(stat%message)
            return
        end if
        call service_tokenizer%open(service_model%file, tokenizer_ok)
        if (.not. tokenizer_ok) then
            write(error_unit, '(a)') 'fortai-native: GGUF tokenizer metadata is unavailable'
            call service_model%close()
            return
        end if
        if (gpu_layers > 0_c_int) then
            call service_model%enable_cuda(int(main_gpu), stat)
            if (.not. stat%is_ok()) then
                write(error_unit, '(a)') 'fortai-native: CUDA initialization failed: ' // trim(stat%message)
                call service_tokenizer%close()
                call service_model%close()
                return
            end if
            if (require_cuda /= 0_c_int .and. .not. service_model%cuda_enabled) then
                write(error_unit, '(a)') 'fortai-native: CUDA was requested but no native CUDA pipeline is active'
                call service_tokenizer%close()
                call service_model%close()
                return
            end if
        end if
        service_ready = .true.
        vocab = int(service_model%vocabulary_size, c_int)
        layers = int(service_model%layer_count, c_int)
        fortai_native_service_init = 0_c_int
    end function fortai_native_service_init

    subroutine fortai_native_service_close() bind(C, name='fortai_native_service_close')
        call service_tokenizer%close()
        call service_model%close()
        service_ready = .false.
    end subroutine fortai_native_service_close

    integer function fortai_native_service_complete_text(prompt_text, max_tokens, output_text, token_count)
        character(len=*), intent(in) :: prompt_text
        integer, intent(in) :: max_tokens
        type(string_t), intent(out) :: output_text
        integer, intent(out) :: token_count
        fortai_native_service_complete_text = fortai_native_service_complete_text_options( &
            prompt_text, max_tokens, 0.0_real32, 0_int64, output_text, token_count)
    end function fortai_native_service_complete_text

    integer function fortai_native_service_complete_text_options(prompt_text, max_tokens, temperature, &
            seed, output_text, token_count)
        character(len=*), intent(in) :: prompt_text
        integer, intent(in) :: max_tokens
        real(real32), intent(in) :: temperature
        integer(int64), intent(in) :: seed
        type(string_t), intent(out) :: output_text
        integer, intent(out) :: token_count
        integer(int32), allocatable :: prompt_ids(:), generated_ids(:)
        real(real32), allocatable :: logits(:)
        integer(int64) :: current, next_token, position
        integer(int64) :: random_state
        integer :: i, generated_count
        real(real32) :: logit_sum
        type(status_t) :: stat

        fortai_native_service_complete_text_options = -1
        token_count = 0
        call output_text%clear()
        if (.not. service_ready .or. max_tokens <= 0 .or. .not. finite_real32(temperature) .or. &
            temperature < 0.0_real32) return
        call service_tokenizer%encode(prompt_text, prompt_ids)
        if (.not. allocated(prompt_ids) .or. size(prompt_ids) == 0 .or. &
            size(prompt_ids) >= service_model%max_context) return
        if (temperature > 0.0_real32) allocate(logits(service_model%vocabulary_size))
        random_state = seed
        if (random_state == 0_int64) then
            block
                integer :: clock
                call system_clock(count=clock)
                random_state = int(clock, int64)
                if (random_state == 0_int64) random_state = int(z'6a09e667f3bcc909', int64)
            end block
        end if
        call service_model%reset()
        current = -1_int64
        do i = 1, size(prompt_ids)
            if (temperature > 0.0_real32) then
                call service_model%forward(int(prompt_ids(i), int64), int(i - 1, int64), logits, stat)
                if (stat%is_ok()) current = sample_logits(logits, temperature, random_state)
            else
                call service_model%forward_greedy(int(prompt_ids(i), int64), int(i - 1, int64), &
                    next_token, logit_sum, stat)
                if (stat%is_ok()) current = next_token
            end if
            if (.not. stat%is_ok()) return
        end do
        allocate(generated_ids(max_tokens))
        generated_count = 0
        do while (generated_count < max_tokens .and. size(prompt_ids) + generated_count < service_model%max_context)
            if (current < 0_int64) exit
            if (service_tokenizer%is_stop(int(current, int32))) exit
            generated_count = generated_count + 1
            generated_ids(generated_count) = int(current, int32)
            if (generated_count == max_tokens) exit
            position = int(size(prompt_ids) + generated_count - 1, int64)
            if (temperature > 0.0_real32) then
                call service_model%forward(current, position, logits, stat)
                if (stat%is_ok()) current = sample_logits(logits, temperature, random_state)
            else
                call service_model%forward_greedy(current, position, next_token, logit_sum, stat)
                if (stat%is_ok()) current = next_token
            end if
            if (.not. stat%is_ok()) return
        end do
        if (generated_count > 0) then
            block
                character(len=:), allocatable :: decoded
                call service_tokenizer%decode(generated_ids(:generated_count), decoded)
                call output_text%set(decoded)
            end block
        end if
        token_count = generated_count
        fortai_native_service_complete_text_options = output_text%length()
    end function fortai_native_service_complete_text_options

    integer(int64) function sample_logits(logits, temperature, random_state)
        real(real32), intent(in) :: logits(:), temperature
        integer(int64), intent(inout) :: random_state
        real(real32) :: maximum, total, threshold, cumulative, probability
        integer :: i, maximum_index

        maximum_index = maxloc(logits, dim=1)
        sample_logits = int(maximum_index - 1, int64)
        maximum = maxval(logits)
        total = 0.0_real32
        do i = 1, size(logits)
            probability = exp((logits(i) - maximum) / temperature)
            total = total + probability
        end do
        if (.not. (total > 0.0_real32)) return
        threshold = uniform_random(random_state) * total
        cumulative = 0.0_real32
        do i = 1, size(logits)
            cumulative = cumulative + exp((logits(i) - maximum) / temperature)
            if (cumulative >= threshold) then
                sample_logits = int(i - 1, int64)
                return
            end if
        end do
    end function sample_logits

    real(real32) function uniform_random(state)
        integer(int64), intent(inout) :: state
        integer(int64), parameter :: mask = int(z'7fffffffffffffff', int64)
        integer(int64) :: bits

        state = ieor(state, ishft(state, 13))
        state = ieor(state, ishft(state, -7))
        state = ieor(state, ishft(state, 17))
        if (state == 0_int64) state = int(z'6a09e667f3bcc909', int64)
        bits = iand(state, mask)
        uniform_random = real(bits, real32) / 9223372036854775808.0_real32
        if (uniform_random <= 0.0_real32) uniform_random = tiny(1.0_real32)
    end function uniform_random

    integer(c_int) function fortai_native_service_complete(prompt, max_tokens, output, capacity, &
            token_count) bind(C, name='fortai_native_service_complete')
        character(kind=c_char), intent(in) :: prompt(*)
        integer(c_int), value, intent(in) :: max_tokens, capacity
        character(kind=c_char), intent(out) :: output(*)
        integer(c_int), intent(out) :: token_count
        character(len=:), allocatable :: prompt_text
        type(string_t) :: output_text
        integer :: result, required

        prompt_text = c_string(prompt)
        result = fortai_native_service_complete_text(prompt_text, int(max_tokens), output_text, required)
        token_count = int(required, c_int)
        if (result < 0) then
            fortai_native_service_complete = -int(max(1, output_text%length()), c_int)
            return
        end if
        if (output_text%length() >= int(capacity)) then
            fortai_native_service_complete = -int(output_text%length(), c_int)
            return
        end if
        call output_text%to_c(output, int(capacity), required)
        fortai_native_service_complete = int(output_text%length(), c_int)
    end function fortai_native_service_complete

end module fortai_native_service
