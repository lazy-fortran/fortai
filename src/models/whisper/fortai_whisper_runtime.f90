module fortai_whisper_runtime
    !! Native Whisper large-v3/turbo execution state.
    !!
    !! This is the model-facing layer: audio is framed, the encoder is run,
    !! cross-attention is prepared, and the decoder is advanced one token at a
    !! time.  No external ASR runtime is called.  The low-level GGML ABI is
    !! used only by the model/encoder/decoder modules below this layer.
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64
    use fortai_string, only: string_t
    use fortai_status, only: FORTAI_INVALID, status_t
    use fortai_whisper_audio, only: whisper_log_mel_spectrogram, whisper_mel_t
    use fortai_whisper_audio_io, only: whisper_wav_read_file
    use fortai_whisper_decoder, only: whisper_decoder_t
    use fortai_whisper_encoder, only: whisper_encoder_t
    use fortai_whisper_model, only: whisper_native_model_t
    use fortai_whisper_tokenizer, only: WHISPER_TOKEN_BEG, WHISPER_TOKEN_EOT, WHISPER_TOKEN_NOT, &
        WHISPER_TOKEN_SOT, WHISPER_TOKEN_TRANSCRIBE, WHISPER_TOKEN_TRANSLATE, whisper_language_token, &
        whisper_token_is_special, whisper_token_piece
    implicit none
    private

    integer(int32), parameter :: WHISPER_CHUNK_SAMPLES = 16000_int32 * 30_int32
    integer(int32), parameter :: WHISPER_DEFAULT_MAX_TOKENS = 256_int32

    type, public :: whisper_native_runtime_t
        type(whisper_native_model_t), pointer :: model => null()
        type(whisper_encoder_t) :: encoder
        type(whisper_decoder_t) :: decoder
        logical :: ready = .false.
        logical :: use_gpu = .false.
        logical :: flash_attention = .true.
        integer(int32) :: gpu_device = 0_int32
        integer(int32) :: threads = 1_int32
    contains
        procedure :: init => whisper_runtime_init
        procedure :: close => whisper_runtime_close
        procedure :: transcribe => whisper_runtime_transcribe
        procedure :: transcribe_wav => whisper_runtime_transcribe_wav
        procedure :: memory_bytes => whisper_runtime_memory_bytes
        procedure :: is_ready => whisper_runtime_is_ready
    end type whisper_native_runtime_t

    public :: whisper_native_argmax

contains

    subroutine whisper_runtime_init(self, path, use_gpu, gpu_device, flash_attention, threads, stat)
        class(whisper_native_runtime_t), intent(inout) :: self
        character(len=*), intent(in) :: path
        logical, intent(in), optional :: use_gpu, flash_attention
        integer(int32), intent(in), optional :: gpu_device, threads
        type(status_t), intent(out) :: stat

        call stat%clear()
        call self%close()
        allocate(self%model)
        self%use_gpu = .false.
        if (present(use_gpu)) self%use_gpu = use_gpu
        self%flash_attention = .true.
        if (present(flash_attention)) self%flash_attention = flash_attention
        self%gpu_device = 0_int32
        if (present(gpu_device)) self%gpu_device = max(0_int32, gpu_device)
        self%threads = 1_int32
        if (present(threads)) self%threads = max(1_int32, threads)

        call self%model%open(trim(path), self%use_gpu, self%gpu_device, stat)
        if (.not. stat%is_ok()) then
            call self%close()
            return
        end if
        call self%encoder%init(self%model, self%flash_attention, stat)
        if (.not. stat%is_ok()) then
            call self%close()
            return
        end if
        self%ready = .true.
    end subroutine whisper_runtime_init

    subroutine whisper_runtime_close(self)
        class(whisper_native_runtime_t), intent(inout) :: self

        call self%decoder%close()
        call self%encoder%close()
        if (associated(self%model)) then
            call self%model%close()
            deallocate(self%model)
        end if
        self%ready = .false.
        self%use_gpu = .false.
        self%flash_attention = .true.
        self%gpu_device = 0_int32
        self%threads = 1_int32
    end subroutine whisper_runtime_close

    subroutine whisper_runtime_transcribe_wav(self, path, text, stat, language, task, max_tokens, temperature, seed)
        class(whisper_native_runtime_t), intent(inout) :: self
        character(len=*), intent(in) :: path
        type(string_t), intent(out) :: text
        type(status_t), intent(out) :: stat
        character(len=*), intent(in), optional :: language, task
        integer(int32), intent(in), optional :: max_tokens
        real(real32), intent(in), optional :: temperature
        integer(int64), intent(in), optional :: seed
        real(real32), allocatable :: samples(:)

        call text%clear()
        call stat%clear()
        call whisper_wav_read_file(path, samples, stat)
        if (.not. stat%is_ok()) return
        call self%transcribe(samples, text, stat, language, task, max_tokens, temperature, seed)
        deallocate(samples)
    end subroutine whisper_runtime_transcribe_wav

    subroutine whisper_runtime_transcribe(self, samples, text, stat, language, task, max_tokens, temperature, seed)
        class(whisper_native_runtime_t), intent(inout) :: self
        real(real32), intent(in) :: samples(:)
        type(string_t), intent(out) :: text
        type(status_t), intent(out) :: stat
        character(len=*), intent(in), optional :: language, task
        integer(int32), intent(in), optional :: max_tokens
        real(real32), intent(in), optional :: temperature
        integer(int64), intent(in), optional :: seed
        integer(int32) :: chunk_start, chunk_count, chunk_end
        real(real32), allocatable :: chunk(:)
        type(string_t) :: chunk_text
        logical :: first_chunk
        integer(int32) :: limit

        call text%clear()
        call stat%clear()
        if (.not. self%is_ready()) then
            call stat%set(FORTAI_INVALID, 'Whisper runtime is not initialized')
            return
        end if
        if (size(samples) <= 0) then
            call stat%set(FORTAI_INVALID, 'Whisper audio payload is empty')
            return
        end if
        limit = WHISPER_DEFAULT_MAX_TOKENS
        if (present(max_tokens)) limit = max(1_int32, min(max_tokens, 4096_int32))
        first_chunk = .true.
        chunk_start = 1_int32
        do while (chunk_start <= size(samples))
            chunk_end = min(int(size(samples), int32), chunk_start + WHISPER_CHUNK_SAMPLES - 1_int32)
            chunk_count = chunk_end - chunk_start + 1_int32
            allocate(chunk(chunk_count))
            chunk = samples(chunk_start:chunk_end)
            call whisper_runtime_transcribe_chunk(self, chunk, chunk_text, stat, language, task, limit, temperature, seed)
            deallocate(chunk)
            if (.not. stat%is_ok()) return
            if (.not. first_chunk .and. chunk_text%length() > 0) call text%append_char(' ')
            call text%append_string(chunk_text)
            first_chunk = .false.
            chunk_start = chunk_end + 1_int32
        end do
    end subroutine whisper_runtime_transcribe

    subroutine whisper_runtime_transcribe_chunk(self, samples, text, stat, language, task, max_tokens, temperature, seed)
        class(whisper_native_runtime_t), intent(inout) :: self
        real(real32), intent(in) :: samples(:)
        type(string_t), intent(out) :: text
        type(status_t), intent(out) :: stat
        character(len=*), intent(in), optional :: language, task
        integer(int32), intent(in) :: max_tokens
        real(real32), intent(in), optional :: temperature
        integer(int64), intent(in), optional :: seed
        type(whisper_mel_t) :: mel
        real(real32), allocatable :: logits(:)
        real(real32), allocatable, target :: encoder_values(:,:)
        type(string_t) :: piece
        integer(int32) :: prefix(4), language_id, position, step, token
        real(real32) :: actual_temperature
        integer(int64) :: random_state
        logical :: valid, emitted

        call text%clear()
        call stat%clear()
        actual_temperature = 0.0_real32
        if (present(temperature)) actual_temperature = max(0.0_real32, temperature)
        random_state = 88172645463325252_int64
        if (present(seed)) then
            if (seed /= 0_int64) random_state = seed
        end if
        language_id = WHISPER_TOKEN_SOT + 1_int32
        if (present(language)) then
            language_id = whisper_language_token(trim(language), stat)
            if (.not. stat%is_ok()) return
        end if
        prefix(1) = WHISPER_TOKEN_SOT
        prefix(2) = language_id
        prefix(3) = WHISPER_TOKEN_TRANSCRIBE
        prefix(4) = WHISPER_TOKEN_NOT
        if (present(task)) then
            if (trim(task) == 'translate') prefix(3) = WHISPER_TOKEN_TRANSLATE
            if (trim(task) /= 'transcribe' .and. trim(task) /= 'translate') then
                call stat%set(FORTAI_INVALID, 'Whisper task must be transcribe or translate')
                return
            end if
        end if

        call whisper_log_mel_spectrogram(samples, self%model%file%filters, mel, self%threads, stat)
        if (.not. stat%is_ok()) return
        call self%encoder%encode(mel, encoder_values, stat)
        if (.not. stat%is_ok()) then
            if (allocated(encoder_values)) deallocate(encoder_values)
            return
        end if
        deallocate(encoder_values)
        call self%decoder%init(self%model, self%flash_attention, stat)
        if (.not. stat%is_ok()) return
        call self%decoder%prepare_cross(self%encoder%output, stat)
        if (.not. stat%is_ok()) return

        position = 0_int32
        do step = 1, size(prefix)
            call self%decoder%decode(prefix(step), position, logits, stat)
            if (.not. stat%is_ok()) return
            position = position + 1_int32
        end do
        do step = 1, max_tokens
            call whisper_suppress_special_tokens(logits, self%model%file%hparams%n_vocab)
            token = whisper_native_sample(logits, actual_temperature, random_state)
            if (token == WHISPER_TOKEN_EOT .or. token == WHISPER_TOKEN_NOT .or. token >= WHISPER_TOKEN_BEG) exit
            if (token < 0_int32 .or. token >= self%model%file%hparams%n_vocab) exit
            call whisper_token_piece(self%model%file, token, piece, valid)
            if (valid) then
                ! Avoid duplicate blank pieces emitted by a malformed model
                ! while preserving the tokenizer's leading-space convention.
                emitted = piece%length() > 0
                if (emitted) call text%append_string(piece)
            end if
            position = position + 1_int32
            if (position >= self%model%file%hparams%n_text_ctx) exit
            call self%decoder%decode(token, position - 1_int32, logits, stat)
            if (.not. stat%is_ok()) return
        end do
    end subroutine whisper_runtime_transcribe_chunk

    subroutine whisper_suppress_special_tokens(logits, n_vocab)
        real(real32), intent(inout) :: logits(:)
        integer(int32), intent(in) :: n_vocab
        integer :: first_special, last_token
        real(real32) :: eot_logit

        first_special = WHISPER_TOKEN_EOT + 1
        last_token = min(size(logits), int(n_vocab, kind=kind(0)))
        if (first_special > last_token) return
        eot_logit = logits(first_special)
        logits(first_special:last_token) = -huge(1.0_real32)
        logits(first_special) = eot_logit
    end subroutine whisper_suppress_special_tokens

    integer(int32) function whisper_native_argmax(logits)
        real(real32), intent(in) :: logits(:)
        integer :: i
        real(real32) :: best

        whisper_native_argmax = -1_int32
        best = -huge(1.0_real32)
        do i = 1, size(logits)
            if (logits(i) > best) then
                best = logits(i)
                whisper_native_argmax = int(i - 1, int32)
            end if
        end do
    end function whisper_native_argmax

    integer(int32) function whisper_native_sample(logits, temperature, state)
        real(real32), intent(in) :: logits(:), temperature
        integer(int64), intent(inout) :: state
        integer :: i, best_index
        real(real64) :: maximum, total, probability, draw, cumulative
        real(real64), allocatable :: probabilities(:)

        if (temperature <= 0.0_real32) then
            whisper_native_sample = whisper_native_argmax(logits)
            return
        end if
        maximum = -huge(1.0_real64)
        do i = 1, size(logits)
            maximum = max(maximum, real(logits(i), real64))
        end do
        allocate(probabilities(size(logits)))
        total = 0.0_real64
        do i = 1, size(logits)
            probabilities(i) = exp((real(logits(i), real64) - maximum) / real(temperature, real64))
            total = total + probabilities(i)
        end do
        if (total <= 0.0_real64) then
            deallocate(probabilities)
            whisper_native_sample = whisper_native_argmax(logits)
            return
        end if
        draw = whisper_native_random(state) * total
        cumulative = 0.0_real64
        best_index = size(logits)
        do i = 1, size(logits)
            cumulative = cumulative + probabilities(i)
            if (draw <= cumulative) then
                best_index = i
                exit
            end if
        end do
        deallocate(probabilities)
        whisper_native_sample = int(best_index - 1, int32)
    end function whisper_native_sample

    real(real64) function whisper_native_random(state)
        integer(int64), intent(inout) :: state
        integer(int64) :: bits

        bits = state
        bits = ieor(bits, ishft(bits, 13))
        bits = ieor(bits, ishft(bits, -7))
        bits = ieor(bits, ishft(bits, 17))
        state = bits
        whisper_native_random = real(iand(bits, int(z'7fffffffffffffff', int64)), real64) / 9223372036854775807.0_real64
    end function whisper_native_random

    integer(int64) function whisper_runtime_memory_bytes(self)
        class(whisper_native_runtime_t), intent(in) :: self

        if (.not. associated(self%model)) then
            whisper_runtime_memory_bytes = 0_int64
            return
        end if
        whisper_runtime_memory_bytes = self%model%memory_bytes() + self%encoder%memory_bytes() + self%decoder%memory_bytes()
    end function whisper_runtime_memory_bytes

    logical function whisper_runtime_is_ready(self)
        class(whisper_native_runtime_t), intent(in) :: self

        whisper_runtime_is_ready = .false.
        if (.not. self%ready) return
        if (.not. self%model%is_ready()) return
        whisper_runtime_is_ready = self%encoder%is_ready()
    end function whisper_runtime_is_ready

end module fortai_whisper_runtime
