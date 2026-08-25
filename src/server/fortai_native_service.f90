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

    type :: sampling_options_t
        integer :: top_k = 20
        real(real32) :: top_p = 0.95_real32
        real(real32) :: min_p = 0.0_real32
        real(real32) :: repeat_penalty = 1.0_real32
        real(real32) :: presence_penalty = 0.0_real32
        real(real32) :: frequency_penalty = 0.0_real32
        integer :: repeat_last_n = 64
    end type sampling_options_t

    public :: fortai_native_service_init
    public :: fortai_native_service_complete
    public :: fortai_native_service_complete_text
    public :: fortai_native_service_complete_text_options
    public :: fortai_native_service_complete_text_sampling
    public :: fortai_native_service_close
    public :: fortai_native_service_default_thinking
    public :: fortai_native_service_supports_reasoning_effort
    public :: fortai_native_service_supports_preserve_thinking

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

    logical function fortai_native_service_default_thinking()
        fortai_native_service_default_thinking = service_tokenizer%default_enable_thinking
    end function fortai_native_service_default_thinking

    logical function fortai_native_service_supports_reasoning_effort()
        fortai_native_service_supports_reasoning_effort = service_tokenizer%supports_reasoning_effort
    end function fortai_native_service_supports_reasoning_effort

    logical function fortai_native_service_supports_preserve_thinking()
        fortai_native_service_supports_preserve_thinking = service_tokenizer%supports_preserve_thinking
    end function fortai_native_service_supports_preserve_thinking

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
        type(sampling_options_t) :: options

        fortai_native_service_complete_text_options = fortai_native_service_complete_text_sampling( &
            prompt_text, max_tokens, temperature, seed, options%top_k, options%top_p, options%min_p, &
            options%repeat_penalty, options%presence_penalty, options%frequency_penalty, &
            options%repeat_last_n, output_text, token_count)
    end function fortai_native_service_complete_text_options

    integer function fortai_native_service_complete_text_sampling(prompt_text, max_tokens, temperature, &
            seed, top_k, top_p, min_p, repeat_penalty, presence_penalty, frequency_penalty, repeat_last_n, &
            output_text, token_count, prompt_token_count)
        character(len=*), intent(in) :: prompt_text
        integer, intent(in) :: max_tokens
        real(real32), intent(in) :: temperature
        integer(int64), intent(in) :: seed
        integer, intent(in) :: top_k, repeat_last_n
        real(real32), intent(in) :: top_p, min_p, repeat_penalty, presence_penalty, frequency_penalty
        type(string_t), intent(out) :: output_text
        integer, intent(out) :: token_count
        integer, intent(out), optional :: prompt_token_count
        type(sampling_options_t) :: options
        integer(int32), allocatable :: prompt_ids(:), generated_ids(:), sampling_history(:)
        integer(int32), allocatable :: history_counts(:), candidate_indices(:)
        real(real32), allocatable :: logits(:), adjusted_logits(:), candidate_values(:)
        integer(int64) :: current, next_token, position
        integer(int64) :: random_state
        integer :: i, generated_count
        real(real32) :: logit_sum
        type(status_t) :: stat

        options%top_k = top_k
        options%top_p = top_p
        options%min_p = min_p
        options%repeat_penalty = repeat_penalty
        options%presence_penalty = presence_penalty
        options%frequency_penalty = frequency_penalty
        options%repeat_last_n = repeat_last_n
        fortai_native_service_complete_text_sampling = -1
        token_count = 0
        if (present(prompt_token_count)) prompt_token_count = 0
        call output_text%clear()
        if (.not. service_ready) return
        if (max_tokens <= 0) return
        if (.not. finite_real32(temperature)) return
        if (temperature < 0.0_real32) return
        if (.not. valid_sampling_options(options)) return
        call service_tokenizer%encode(prompt_text, prompt_ids)
        if (.not. allocated(prompt_ids)) then
            write(error_unit, '(a)') 'fortai-native: prompt tokenization produced no buffer'
            return
        end if
        if (size(prompt_ids) == 0) then
            write(error_unit, '(a)') 'fortai-native: prompt tokenization produced no tokens'
            return
        end if
        if (present(prompt_token_count)) prompt_token_count = size(prompt_ids)
        if (size(prompt_ids) >= service_model%max_context) then
            write(error_unit, '(a,i0,a,i0,a)') 'fortai-native: prompt has ', size(prompt_ids), &
                ' tokens but context is ', service_model%max_context, ' tokens'
            return
        end if
        allocate(generated_ids(max_tokens))
        allocate(sampling_history(size(prompt_ids) + max_tokens))
        sampling_history(:size(prompt_ids)) = prompt_ids
        if (temperature > 0.0_real32) then
            allocate(logits(service_model%vocabulary_size))
            allocate(adjusted_logits(service_model%vocabulary_size))
            allocate(history_counts(service_model%vocabulary_size))
            allocate(candidate_indices(service_model%vocabulary_size))
            allocate(candidate_values(service_model%vocabulary_size))
        else if (sampling_penalties_active(options)) then
            allocate(logits(service_model%vocabulary_size))
            allocate(adjusted_logits(service_model%vocabulary_size))
            allocate(history_counts(service_model%vocabulary_size))
        end if
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
                if (stat%is_ok()) current = sample_logits(logits, temperature, random_state, sampling_history, &
                    i, options, adjusted_logits, history_counts, candidate_indices, candidate_values)
            else
                if (sampling_penalties_active(options)) then
                    call service_model%forward(int(prompt_ids(i), int64), int(i - 1, int64), logits, stat)
                    if (stat%is_ok()) then
                        current = greedy_penalized(logits, adjusted_logits, history_counts, sampling_history, i, options)
                    end if
                else
                    call service_model%forward_greedy(int(prompt_ids(i), int64), int(i - 1, int64), &
                        next_token, logit_sum, stat)
                    if (stat%is_ok()) current = next_token
                end if
            end if
            if (.not. stat%is_ok()) then
                write(error_unit, '(a)') 'fortai-native: model forward failed: ' // trim(stat%message)
                return
            end if
        end do
        generated_count = 0
        do while (generated_count < max_tokens .and. size(prompt_ids) + generated_count < service_model%max_context)
            if (current < 0_int64) exit
            if (service_tokenizer%is_stop(int(current, int32))) exit
            generated_count = generated_count + 1
            generated_ids(generated_count) = int(current, int32)
            sampling_history(size(prompt_ids) + generated_count) = int(current, int32)
            if (generated_count == max_tokens) exit
            position = int(size(prompt_ids) + generated_count - 1, int64)
            if (temperature > 0.0_real32) then
                call service_model%forward(current, position, logits, stat)
                if (stat%is_ok()) current = sample_logits(logits, temperature, random_state, sampling_history, &
                    size(prompt_ids) + generated_count, options, adjusted_logits, history_counts, candidate_indices, &
                    candidate_values)
            else
                if (sampling_penalties_active(options)) then
                    call service_model%forward(current, position, logits, stat)
                    if (stat%is_ok()) then
                        current = greedy_penalized(logits, adjusted_logits, history_counts, sampling_history, &
                            size(prompt_ids) + generated_count, options)
                    end if
                else
                    call service_model%forward_greedy(current, position, next_token, logit_sum, stat)
                    if (stat%is_ok()) current = next_token
                end if
            end if
            if (.not. stat%is_ok()) then
                write(error_unit, '(a)') 'fortai-native: model forward failed: ' // trim(stat%message)
                return
            end if
        end do
        if (generated_count > 0) then
            block
                character(len=:), allocatable :: decoded
                call service_tokenizer%decode(generated_ids(:generated_count), decoded)
                call output_text%set(decoded)
            end block
        end if
        token_count = generated_count
        fortai_native_service_complete_text_sampling = output_text%length()
    end function fortai_native_service_complete_text_sampling

    logical function valid_sampling_options(options)
        type(sampling_options_t), intent(in) :: options

        valid_sampling_options = .false.
        if (options%top_k < 0 .or. options%repeat_last_n < -1) return
        if (.not. finite_real32(options%top_p)) return
        if (.not. finite_real32(options%min_p)) return
        if (.not. finite_real32(options%repeat_penalty)) return
        if (.not. finite_real32(options%presence_penalty)) return
        if (.not. finite_real32(options%frequency_penalty)) return
        if (options%top_p <= 0.0_real32 .or. options%top_p > 1.0_real32) return
        if (options%min_p < 0.0_real32 .or. options%min_p > 1.0_real32) return
        if (options%repeat_penalty <= 0.0_real32) return
        valid_sampling_options = .true.
    end function valid_sampling_options

    logical function sampling_penalties_active(options)
        type(sampling_options_t), intent(in) :: options

        sampling_penalties_active = .false.
        if (options%repeat_last_n /= 0) then
            if (options%repeat_penalty /= 1.0_real32) sampling_penalties_active = .true.
            if (options%presence_penalty /= 0.0_real32) sampling_penalties_active = .true.
            if (options%frequency_penalty /= 0.0_real32) sampling_penalties_active = .true.
        end if
    end function sampling_penalties_active

    integer(int64) function greedy_penalized(logits, adjusted, counts, history, history_count, options)
        real(real32), intent(in) :: logits(:)
        real(real32), intent(out) :: adjusted(:)
        integer(int32), intent(inout) :: counts(:)
        integer(int32), intent(in) :: history(:)
        integer, intent(in) :: history_count
        type(sampling_options_t), intent(in) :: options
        integer :: maximum_index

        call apply_sampling_penalties(logits, adjusted, counts, history, history_count, options)
        maximum_index = maxloc(adjusted, dim=1)
        greedy_penalized = int(maximum_index - 1, int64)
    end function greedy_penalized

    integer(int64) function sample_logits(logits, temperature, random_state, history, history_count, options, &
            adjusted, counts, candidate_indices, candidate_values)
        real(real32), intent(in) :: logits(:), temperature
        integer(int64), intent(inout) :: random_state
        integer(int32), intent(in) :: history(:)
        integer, intent(in) :: history_count
        type(sampling_options_t), intent(in) :: options
        real(real32), intent(inout) :: adjusted(:), candidate_values(:)
        integer(int32), intent(inout) :: counts(:), candidate_indices(:)
        real(real32) :: maximum, total, threshold, cumulative, probability, cutoff
        integer :: i, candidate_count, selected_count

        call apply_sampling_penalties(logits, adjusted, counts, history, history_count, options)
        candidate_count = min(size(adjusted), size(candidate_indices))
        if (options%top_k > 0) candidate_count = min(candidate_count, options%top_k)
        call top_candidates(adjusted, candidate_count, candidate_indices, candidate_values)
        maximum = candidate_values(1)
        total = 0.0_real32
        cutoff = options%min_p
        do i = 1, candidate_count
            probability = exp((candidate_values(i) - maximum) / temperature)
            if (probability < cutoff) then
                candidate_values(i) = 0.0_real32
            else
                candidate_values(i) = probability
                total = total + probability
            end if
        end do
        if (.not. (total > 0.0_real32)) then
            sample_logits = int(candidate_indices(1) - 1, int64)
            return
        end if
        selected_count = candidate_count
        if (options%top_p < 1.0_real32) then
            cutoff = total * options%top_p
            cumulative = 0.0_real32
            selected_count = 1
            do i = 1, candidate_count
                if (candidate_values(i) > 0.0_real32) cumulative = cumulative + candidate_values(i)
                if (cumulative >= cutoff) then
                    selected_count = i
                    exit
                end if
            end do
        end if
        threshold = uniform_random(random_state) * total_probability(candidate_values, selected_count)
        cumulative = 0.0_real32
        sample_logits = int(candidate_indices(1) - 1, int64)
        do i = 1, selected_count
            if (candidate_values(i) <= 0.0_real32) cycle
            cumulative = cumulative + candidate_values(i)
            if (cumulative >= threshold) then
                sample_logits = int(candidate_indices(i) - 1, int64)
                return
            end if
        end do
    end function sample_logits

    subroutine apply_sampling_penalties(logits, adjusted, counts, history, history_count, options)
        real(real32), intent(in) :: logits(:)
        real(real32), intent(out) :: adjusted(:)
        integer(int32), intent(inout) :: counts(:)
        integer(int32), intent(in) :: history(:)
        integer, intent(in) :: history_count
        type(sampling_options_t), intent(in) :: options
        integer :: i, first, last, token

        adjusted = logits
        if (size(counts) > 0) counts = 0_int32
        if (options%repeat_last_n == 0 .or. history_count <= 0) return
        if (options%repeat_last_n < 0) then
            first = 1
        else
            first = max(1, history_count - options%repeat_last_n + 1)
        end if
        last = min(history_count, size(history))
        if (first > last) return
        do i = first, last
            token = int(history(i)) + 1
            if (token >= 1 .and. token <= size(counts)) counts(token) = counts(token) + 1_int32
        end do
        do i = 1, size(adjusted)
            if (counts(i) <= 0) cycle
            if (options%repeat_penalty /= 1.0_real32) then
                if (adjusted(i) > 0.0_real32) then
                    adjusted(i) = adjusted(i) / options%repeat_penalty
                else
                    adjusted(i) = adjusted(i) * options%repeat_penalty
                end if
            end if
            adjusted(i) = adjusted(i) - options%presence_penalty
            adjusted(i) = adjusted(i) - options%frequency_penalty * real(counts(i), real32)
        end do
    end subroutine apply_sampling_penalties

    real(real32) function total_probability(values, count)
        real(real32), intent(in) :: values(:)
        integer, intent(in) :: count
        integer :: i

        total_probability = 0.0_real32
        do i = 1, count
            if (values(i) > 0.0_real32) total_probability = total_probability + values(i)
        end do
    end function total_probability

    subroutine top_candidates(values, count, indices, selected)
        real(real32), intent(in) :: values(:)
        integer, intent(in) :: count
        integer(int32), intent(out) :: indices(:)
        real(real32), intent(out) :: selected(:)
        integer :: i, heap_count

        heap_count = min(count, size(indices))
        do i = 1, heap_count
            indices(i) = int(i, int32)
            selected(i) = values(i)
        end do
        if (heap_count <= 0) return
        do i = heap_count / 2, 1, -1
            call sift_min_heap(selected, indices, heap_count, i)
        end do
        do i = heap_count + 1, size(values)
            if (comes_before(values(i), int(i, int32), selected(1), indices(1))) then
                selected(1) = values(i)
                indices(1) = int(i, int32)
                call sift_min_heap(selected, indices, heap_count, 1)
            end if
        end do
        call sort_descending(selected, indices, heap_count)
    end subroutine top_candidates

    logical function comes_before(left_value, left_index, right_value, right_index)
        real(real32), intent(in) :: left_value, right_value
        integer(int32), intent(in) :: left_index, right_index

        comes_before = left_value > right_value
        if (left_value == right_value) comes_before = left_index < right_index
    end function comes_before

    logical function comes_before_ascending(left_value, left_index, right_value, right_index)
        real(real32), intent(in) :: left_value, right_value
        integer(int32), intent(in) :: left_index, right_index

        comes_before_ascending = left_value < right_value
        if (left_value == right_value) comes_before_ascending = left_index < right_index
    end function comes_before_ascending

    subroutine sift_min_heap(values, indices, count, root)
        real(real32), intent(inout) :: values(:)
        integer(int32), intent(inout) :: indices(:)
        integer, intent(in) :: count, root
        integer :: parent, child
        real(real32) :: value
        integer(int32) :: index

        parent = root
        value = values(parent)
        index = indices(parent)
        do
            child = 2 * parent
            if (child > count) exit
            if (child + 1 <= count) then
                if (comes_before_ascending(values(child + 1), indices(child + 1), values(child), indices(child))) then
                    child = child + 1
                end if
            end if
            if (.not. comes_before_ascending(values(child), indices(child), value, index)) exit
            values(parent) = values(child)
            indices(parent) = indices(child)
            parent = child
        end do
        values(parent) = value
        indices(parent) = index
    end subroutine sift_min_heap

    subroutine sort_descending(values, indices, count)
        real(real32), intent(inout) :: values(:)
        integer(int32), intent(inout) :: indices(:)
        integer, intent(in) :: count
        integer :: i
        real(real32) :: value
        integer(int32) :: index

        do i = count / 2, 1, -1
            call sift_max_heap(values, indices, count, i)
        end do
        do i = count, 2, -1
            value = values(1); values(1) = values(i); values(i) = value
            index = indices(1); indices(1) = indices(i); indices(i) = index
            call sift_max_heap(values, indices, i - 1, 1)
        end do
        do i = 1, count / 2
            value = values(i); values(i) = values(count - i + 1); values(count - i + 1) = value
            index = indices(i); indices(i) = indices(count - i + 1); indices(count - i + 1) = index
        end do
    end subroutine sort_descending

    subroutine sift_max_heap(values, indices, count, root)
        real(real32), intent(inout) :: values(:)
        integer(int32), intent(inout) :: indices(:)
        integer, intent(in) :: count, root
        integer :: parent, child
        real(real32) :: value
        integer(int32) :: index

        parent = root
        value = values(parent)
        index = indices(parent)
        do
            child = 2 * parent
            if (child > count) exit
            if (child + 1 <= count) then
                if (comes_before(values(child + 1), indices(child + 1), values(child), indices(child))) child = child + 1
            end if
            if (.not. comes_before(values(child), indices(child), value, index)) exit
            values(parent) = values(child)
            indices(parent) = indices(child)
            parent = child
        end do
        values(parent) = value
        indices(parent) = index
    end subroutine sift_max_heap

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
