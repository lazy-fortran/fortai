module fortai_native_service
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char
    use, intrinsic :: iso_fortran_env, only: error_unit, int32, int64, real32
    use fortai_gguf_runtime, only: gguf_file_t
    use fortai_native_tokenizer, only: fortai_native_tokenizer_t
    use fortai_qwen35_cpu, only: qwen35_cpu_model_t
    use fortai_string, only: string_t
    use fortai_status, only: FORTAI_INVALID, FORTAI_UNSUPPORTED, status_t
    implicit none
    private

    type(qwen35_cpu_model_t), save :: service_model
    type(fortai_native_tokenizer_t), save :: service_tokenizer
    type(qwen35_cpu_model_t), save :: service_draft_model
    type(fortai_native_tokenizer_t), save :: service_draft_tokenizer
    ! MTP tensors are rebound into service_model.  Keep the sidecar mapping
    ! alive for exactly as long as those pointers are in use.
    type(gguf_file_t), save :: service_mtp_sidecar
    ! The cache keeps the exact token prefix represented by the resident model
    ! state.  It is deliberately single-slot: CUDA is serialized by the HTTP
    ! service, so retaining the live KV/recurrent state avoids a second model
    ! copy while still making the common OpenCode conversation prefix reusable.
    integer(int32), allocatable, save :: service_cache_tokens(:)
    real(real32), allocatable, save :: service_cache_logits(:)
    integer, save :: service_cache_count = 0
    integer(int64), save :: service_cache_next_token = -1_int64
    logical, save :: service_cache_valid = .false.
    logical, save :: service_cache_logits_valid = .false.
    logical, save :: service_ready = .false.
    logical, save :: service_external_draft_active = .false.
    logical, save :: service_mtp_sidecar_active = .false.

    type :: sampling_options_t
        integer :: top_k = 40
        real(real32) :: top_p = 0.95_real32
        real(real32) :: min_p = 0.05_real32
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
    public :: fortai_native_service_tokenize
    public :: fortai_native_service_detokenize
    public :: fortai_native_service_token_piece
    public :: fortai_native_service_close
    public :: fortai_native_service_default_thinking
    public :: fortai_native_service_supports_reasoning_effort
    public :: fortai_native_service_supports_preserve_thinking
    public :: fortai_native_service_mtp_available
    public :: fortai_native_service_mtp_active
    public :: fortai_native_service_external_draft_active
    public :: fortai_native_service_mtp_sidecar_active
    public :: fortai_native_service_device_pipeline
    public :: fortai_native_service_context_size
    public :: fortai_native_service_cache_reuse_supported
    public :: fortai_native_service_cache_reuse_active
    public :: fortai_native_service_cache_reuse_count

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
        character(len=:), allocatable :: draft_path

        fortai_native_service_init = 1_c_int
        vocab = 0_c_int
        layers = 0_c_int
        call fortai_native_service_close()
        model_path = c_string(path)
        if (len_trim(model_path) == 0 .or. context_size < 0 .or. threads <= 0) return
        call service_model%open(trim(model_path), int(context_size, int64), stat)
        if (.not. stat%is_ok()) then
            write(error_unit, '(a)') 'fortai-native: model open failed: ' // trim(stat%message)
            return
        end if
        draft_path = configured_draft_path()
        if (service_model%mtp_active .and. is_mtp_sidecar_path(draft_path)) then
            call apply_mtp_sidecar(draft_path, stat)
            if (.not. stat%is_ok()) then
                write(error_unit, '(a)') 'fortai-native: MTP sidecar initialization failed: ' // trim(stat%message)
                call service_model%close()
                return
            end if
        end if
        call service_tokenizer%open(service_model%file, tokenizer_ok)
        if (.not. tokenizer_ok) then
            write(error_unit, '(a)') 'fortai-native: GGUF tokenizer metadata is unavailable'
            call service_model%close()
            call service_mtp_sidecar%close()
            return
        end if
        draft_path = configured_external_draft_path()
        call open_external_draft(draft_path, context_size, stat)
        if (.not. stat%is_ok()) then
            write(error_unit, '(a)') 'fortai-native: external draft initialization failed: ' // trim(stat%message)
            call service_tokenizer%close()
            call service_model%close()
            call service_mtp_sidecar%close()
            return
        end if
        if (gpu_layers > 0_c_int) then
            call service_model%enable_cuda(int(main_gpu), stat)
            if (.not. stat%is_ok()) then
                write(error_unit, '(a)') 'fortai-native: CUDA initialization failed: ' // trim(stat%message)
                call service_draft_tokenizer%close()
                call service_draft_model%close()
                call service_tokenizer%close()
                call service_model%close()
                call service_mtp_sidecar%close()
                return
            end if
            if (require_cuda /= 0_c_int .and. .not. service_model%cuda_enabled) then
                write(error_unit, '(a)') 'fortai-native: CUDA was requested but no native CUDA pipeline is active'
                call service_draft_tokenizer%close()
                call service_draft_model%close()
                call service_tokenizer%close()
                call service_model%close()
                call service_mtp_sidecar%close()
                return
            end if
        end if
        service_ready = .true.
        vocab = int(service_model%vocabulary_size, c_int)
        layers = int(service_model%layer_count, c_int)
        fortai_native_service_init = 0_c_int
    end function fortai_native_service_init

    subroutine fortai_native_service_close() bind(C, name='fortai_native_service_close')
        call service_draft_tokenizer%close()
        call service_draft_model%close()
        call service_tokenizer%close()
        call service_model%close()
        ! The model owns pointers into this mapping for MTP tensors, so it
        ! must be closed before releasing the sidecar address space.
        call service_mtp_sidecar%close()
        if (allocated(service_cache_tokens)) deallocate(service_cache_tokens)
        if (allocated(service_cache_logits)) deallocate(service_cache_logits)
        service_cache_count = 0
        service_cache_next_token = -1_int64
        service_cache_valid = .false.
        service_cache_logits_valid = .false.
        service_ready = .false.
        service_external_draft_active = .false.
        service_mtp_sidecar_active = .false.
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

    logical function fortai_native_service_mtp_available()
        fortai_native_service_mtp_available = service_ready .and. service_model%mtp_available
    end function fortai_native_service_mtp_available

    logical function fortai_native_service_mtp_active()
        fortai_native_service_mtp_active = service_ready .and. service_model%mtp_active
    end function fortai_native_service_mtp_active

    logical function fortai_native_service_external_draft_active()
        fortai_native_service_external_draft_active = service_ready .and. service_external_draft_active
    end function fortai_native_service_external_draft_active

    logical function fortai_native_service_mtp_sidecar_active()
        fortai_native_service_mtp_sidecar_active = service_ready .and. service_mtp_sidecar_active
    end function fortai_native_service_mtp_sidecar_active

    logical function fortai_native_service_device_pipeline()
        fortai_native_service_device_pipeline = service_ready .and. service_model%cuda_device_pipeline
    end function fortai_native_service_device_pipeline

    integer(int64) function fortai_native_service_context_size()
        fortai_native_service_context_size = 0_int64
        if (service_ready) fortai_native_service_context_size = service_model%max_context
    end function fortai_native_service_context_size

    logical function fortai_native_service_cache_reuse_supported()
        ! The live model state is retained between serialized requests.  This
        ! is a native prefix-KV cache, not a delegated runtime cache.
        fortai_native_service_cache_reuse_supported = .true.
    end function fortai_native_service_cache_reuse_supported

    logical function fortai_native_service_cache_reuse_active()
        fortai_native_service_cache_reuse_active = .false.
        if (.not. service_ready) return
        if (.not. service_cache_valid) return
        fortai_native_service_cache_reuse_active = service_cache_count > 0
    end function fortai_native_service_cache_reuse_active

    integer(int64) function fortai_native_service_cache_reuse_count()
        fortai_native_service_cache_reuse_count = 0_int64
        if (service_cache_valid) fortai_native_service_cache_reuse_count = int(service_cache_count, int64)
    end function fortai_native_service_cache_reuse_count

    logical function fortai_native_service_tokenize(text, add_special, parse_special, ids)
        character(len=*), intent(in) :: text
        logical, intent(in) :: add_special, parse_special
        integer(int32), allocatable, intent(out) :: ids(:)

        allocate(ids(0))
        fortai_native_service_tokenize = .false.
        if (.not. service_ready) return
        call service_tokenizer%encode(text, ids, add_special, parse_special)
        fortai_native_service_tokenize = allocated(ids)
    end function fortai_native_service_tokenize

    logical function fortai_native_service_detokenize(ids, text)
        integer(int32), intent(in) :: ids(:)
        character(len=:), allocatable, intent(out) :: text
        integer :: i

        fortai_native_service_detokenize = .false.
        if (.not. service_ready) then
            allocate(character(len=0) :: text)
            return
        end if
        do i = 1, size(ids)
            if (ids(i) < 0_int32 .or. ids(i) >= service_tokenizer%vocab_size) then
                allocate(character(len=0) :: text)
                return
            end if
        end do
        call service_tokenizer%decode(ids, text)
        fortai_native_service_detokenize = allocated(text)
    end function fortai_native_service_detokenize

    logical function fortai_native_service_token_piece(token, piece)
        integer(int32), intent(in) :: token
        character(len=:), allocatable, intent(out) :: piece
        logical :: valid

        allocate(character(len=0) :: piece)
        fortai_native_service_token_piece = .false.
        if (.not. service_ready) return
        call service_tokenizer%token_piece(token, piece, valid)
        fortai_native_service_token_piece = valid
    end function fortai_native_service_token_piece

    logical function native_mtp_mode_requested()
        character(len=64) :: value
        character(len=:), allocatable :: draft_path
        integer :: length

        native_mtp_mode_requested = .false.
        value = ''
        call get_environment_variable('FORTAI_NATIVE_MTP', value, length=length)
        if (length > 0) then
            length = min(length, len(value))
            if (trim(value(:length)) == '1') then
                native_mtp_mode_requested = .true.
                return
            end if
            if (trim(value(:length)) == 'true') then
                native_mtp_mode_requested = .true.
                return
            end if
            if (trim(value(:length)) == 'on') then
                native_mtp_mode_requested = .true.
                return
            end if
        end if
        value = ''
        call get_environment_variable('FORTAI_SPEC_TYPE', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMA_ARG_SPEC_TYPE', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_SPEC_TYPE', value, length=length)
        if (length > 0) then
            length = min(length, len(value))
            native_mtp_mode_requested = trim(value(:length)) == 'draft-mtp'
        end if
        if (.not. native_mtp_mode_requested) then
            ! The production llama launcher selects MTP from a sidecar draft
            ! path. Preserve that drop-in behavior when the native server is
            ! launched directly and no explicit --spec-type was exported.
            draft_path = configured_draft_path()
            native_mtp_mode_requested = is_mtp_sidecar_path(draft_path)
        end if
    end function native_mtp_mode_requested

    logical function native_ignore_eos_requested()
        character(len=32) :: value
        integer :: length

        native_ignore_eos_requested = .false.
        value = ''
        call get_environment_variable('FORTAI_IGNORE_EOS', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMA_ARG_IGNORE_EOS', value, length=length)
        if (length <= 0 .or. length > len(value)) return
        select case (trim(value(:length)))
        case ('1', 'true', 'on', 'yes')
            native_ignore_eos_requested = .true.
        end select
    end function native_ignore_eos_requested

    function configured_external_draft_path() result(path)
        character(len=:), allocatable :: path
        character(len=:), allocatable :: configured

        configured = configured_draft_path()
        if (len_trim(configured) == 0) then
            allocate(character(len=0) :: path)
        else if (native_mtp_mode_requested() .or. is_mtp_sidecar_path(configured)) then
            allocate(character(len=0) :: path)
        else
            allocate(character(len=len_trim(configured)) :: path)
            path = trim(configured)
        end if
    end function configured_external_draft_path

    function configured_draft_path() result(path)
        character(len=:), allocatable :: path
        character(len=4096) :: value
        integer :: length

        value = ''
        call get_environment_variable('FORTAI_DRAFT_MODEL', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMA_ARG_MODEL_DRAFT', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_DRAFT_MODEL', value, length=length)
        if (length <= 0) then
            allocate(character(len=0) :: path)
            return
        end if
        length = min(length, len(value))
        allocate(character(len=len_trim(value(:length))) :: path)
        if (len(path) > 0) path = trim(value(:length))
    end function configured_draft_path

    logical function is_mtp_sidecar_path(path)
        character(len=*), intent(in) :: path

        is_mtp_sidecar_path = .false.
        if (len_trim(path) == 0) return
        is_mtp_sidecar_path = index(path, 'mtp') > 0 .or. index(path, 'MTP') > 0
    end function is_mtp_sidecar_path

    subroutine apply_mtp_sidecar(path, stat)
        character(len=*), intent(in) :: path
        type(status_t), intent(out) :: stat
        integer :: i, target_index, copied
        character(len=:), allocatable :: tensor_name

        call stat%clear()
        service_mtp_sidecar_active = .false.
        if (len_trim(path) == 0) return
        ! Keep the sidecar mapping alive after transferring the selected
        ! blk.64 tensor pointers into the target model.  Copying this ~1.3 GiB
        ! GGUF is needlessly expensive at startup and raises peak RSS.
        call service_mtp_sidecar%open(trim(path), stat, .true.)
        if (.not. stat%is_ok()) return
        if (.not. allocated(service_mtp_sidecar%tensors)) then
            call stat%set(FORTAI_INVALID, 'MTP sidecar has no tensors')
            call service_mtp_sidecar%close()
            return
        end if
        copied = 0
        do i = 1, size(service_mtp_sidecar%tensors)
            tensor_name = service_mtp_sidecar%tensors(i)%name
            if (index(tensor_name, 'blk.64.') /= 1) cycle
            target_index = service_model%file%find_tensor(tensor_name)
            if (target_index == 0) then
                call stat%set(FORTAI_UNSUPPORTED, 'MTP sidecar tensor is absent from target: ' // trim(tensor_name))
                exit
            end if
            if (.not. allocated(service_mtp_sidecar%tensors(i)%shape)) then
                call stat%set(FORTAI_INVALID, 'MTP sidecar tensor has no shape: ' // trim(tensor_name))
                exit
            end if
            if (.not. allocated(service_model%file%tensors(target_index)%shape)) then
                call stat%set(FORTAI_INVALID, 'target MTP tensor has no shape: ' // trim(tensor_name))
                exit
            end if
            if (size(service_mtp_sidecar%tensors(i)%shape) /= size(service_model%file%tensors(target_index)%shape)) then
                call stat%set(FORTAI_INVALID, 'MTP sidecar tensor rank differs from target: ' // trim(tensor_name))
                exit
            end if
            if (any(service_mtp_sidecar%tensors(i)%shape /= service_model%file%tensors(target_index)%shape)) then
                call stat%set(FORTAI_INVALID, 'MTP sidecar tensor shape differs from target: ' // trim(tensor_name))
                exit
            end if
            if (.not. associated(service_mtp_sidecar%tensors(i)%bytes)) then
                call stat%set(FORTAI_INVALID, 'MTP sidecar tensor has no data: ' // trim(tensor_name))
                exit
            end if
            if (service_mtp_sidecar%tensors(i)%byte_count /= &
                int(size(service_mtp_sidecar%tensors(i)%bytes), int64)) then
                call stat%set(FORTAI_INVALID, 'MTP sidecar tensor byte count is inconsistent: ' // trim(tensor_name))
                exit
            end if
            if (associated(service_model%file%tensors(target_index)%bytes)) then
                if (service_model%file%tensors(target_index)%bytes_mapped) then
                    nullify(service_model%file%tensors(target_index)%bytes)
                else
                    deallocate(service_model%file%tensors(target_index)%bytes)
                end if
            end if
            service_model%file%tensors(target_index)%bytes => service_mtp_sidecar%tensors(i)%bytes
            service_model%file%tensors(target_index)%bytes_mapped = .true.
            service_model%file%tensors(target_index)%bytes_mapped_external = .true.
            nullify(service_mtp_sidecar%tensors(i)%bytes)
            service_mtp_sidecar%tensors(i)%bytes_mapped = .false.
            service_mtp_sidecar%tensors(i)%bytes_mapped_external = .false.
            service_model%file%tensors(target_index)%value_type = service_mtp_sidecar%tensors(i)%value_type
            service_model%file%tensors(target_index)%byte_count = service_mtp_sidecar%tensors(i)%byte_count
            if (allocated(service_model%file%tensors(target_index)%decoded_values)) &
                deallocate(service_model%file%tensors(target_index)%decoded_values)
            if (allocated(service_mtp_sidecar%tensors(i)%decoded_values)) then
                call move_alloc(service_mtp_sidecar%tensors(i)%decoded_values, &
                    service_model%file%tensors(target_index)%decoded_values)
            end if
            copied = copied + 1
        end do
        if (stat%is_ok() .and. copied < 14) then
            call stat%set(FORTAI_UNSUPPORTED, 'MTP sidecar does not contain the complete blk.64 head')
        end if
        if (.not. stat%is_ok()) then
            call service_mtp_sidecar%close()
            return
        end if
        service_mtp_sidecar_active = .true.
        write(error_unit, '(a,i0,a)') 'fortai-native: loaded ', copied, ' MTP head tensors from sidecar'
    end subroutine apply_mtp_sidecar

    subroutine open_external_draft(path, context_size, stat)
        character(len=*), intent(in) :: path
        integer(c_int), intent(in) :: context_size
        type(status_t), intent(out) :: stat
        logical :: tokenizer_ok
        character(len=:), allocatable :: draft_error

        call stat%clear()
        service_external_draft_active = .false.
        if (len_trim(path) == 0) return
        call service_draft_model%open(trim(path), int(context_size, int64), stat)
        if (.not. stat%is_ok()) then
            draft_error = ''
            if (allocated(stat%message)) draft_error = stat%message
            call stat%set(FORTAI_UNSUPPORTED, 'external draft model open failed: ' // trim(draft_error))
            return
        end if
        if (service_draft_model%vocabulary_size /= service_model%vocabulary_size) then
            call stat%set(FORTAI_UNSUPPORTED, 'external draft vocabulary does not match target model')
            call service_draft_model%close()
            return
        end if
        call service_draft_tokenizer%open(service_draft_model%file, tokenizer_ok)
        if (.not. tokenizer_ok) then
            call stat%set(FORTAI_UNSUPPORTED, 'external draft tokenizer metadata is unavailable')
            call service_draft_model%close()
            return
        end if
        service_external_draft_active = .true.
    end subroutine open_external_draft

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
        integer(int32), allocatable :: prompt_ids(:), generated_ids(:), sampling_history(:), state_tokens(:)
        integer(int32), allocatable :: history_counts(:), candidate_indices(:)
        real(real32), allocatable :: logits(:), adjusted_logits(:), candidate_values(:)
        integer(int64) :: current, next_token, position
        integer(int64) :: draft_next_token
        integer(int64) :: speculative_tokens(32)
        integer(int64) :: random_state
        integer :: i, j, generated_count, speculative_count, prompt_start, state_count, cache_count
        real(real32) :: logit_sum, draft_logit_sum
        logical :: use_external_draft, use_native_mtp
        logical :: stop_generation, ignore_eos, cache_hit, cache_logits_usable
        logical :: needs_logits, preserve_old_logits, state_step_speculative
        integer(int32), allocatable :: new_cache_tokens(:)
        real(real32), allocatable :: new_cache_logits(:)
        type(status_t) :: stat, draft_stat

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
        allocate(state_tokens(size(prompt_ids) + max_tokens))
        sampling_history(:size(prompt_ids)) = prompt_ids
        state_tokens(:size(prompt_ids)) = prompt_ids
        needs_logits = temperature > 0.0_real32
        if (.not. needs_logits) needs_logits = sampling_penalties_active(options)
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
        ! Keep the standalone draft state synchronized with the target.  The
        ! target remains authoritative because the native hybrid model has no
        ! multi-token decode ABI yet; this still makes external-draft wiring
        ! real and deterministic rather than merely accepting the flag.
        use_external_draft = service_external_draft_active .and. temperature == 0.0_real32 .and. &
            .not. sampling_penalties_active(options)
        use_native_mtp = service_model%mtp_active .and. temperature == 0.0_real32 .and. &
            .not. sampling_penalties_active(options)
        ignore_eos = native_ignore_eos_requested()
        cache_count = service_cache_count
        cache_hit = service_cache_valid
        if (cache_hit) then
            if (cache_count <= 0) cache_hit = .false.
        end if
        if (cache_hit) then
            if (.not. allocated(service_cache_tokens)) cache_hit = .false.
        end if
        if (cache_hit) then
            if (size(prompt_ids) < cache_count) cache_hit = .false.
        end if
        if (cache_hit) then
            do i = 1, cache_count
                if (prompt_ids(i) /= service_cache_tokens(i)) then
                    cache_hit = .false.
                    exit
                end if
            end do
        end if
        ! A standalone external draft has its own recurrent/KV state.  Until
        ! that second state is retained atomically, replay the request rather
        ! than reusing only the target prefix.
        if (cache_hit) then
            if (service_external_draft_active) cache_hit = .false.
        end if
        cache_logits_usable = .false.
        if (cache_hit) then
            if (service_cache_logits_valid) then
                if (allocated(service_cache_logits)) then
                    if (size(service_cache_logits) == service_model%vocabulary_size) cache_logits_usable = .true.
                end if
            end if
            if (needs_logits) then
                if (.not. cache_logits_usable) cache_hit = .false.
            end if
        end if
        prompt_start = 1
        state_count = 0
        if (cache_hit) then
            prompt_start = cache_count + 1
            state_count = cache_count
            ! Mark the old entry stale while this request mutates the live
            ! state.  Any forward failure therefore cannot leave a false hit.
            service_cache_valid = .false.
        else
            call service_model%reset()
            service_cache_valid = .false.
        end if
        if (use_external_draft) call service_draft_model%reset()
        current = -1_int64
        call stat%clear()
        if (cache_hit) then
            if (prompt_start > size(prompt_ids)) then
                if (cache_logits_usable) then
                    if (temperature > 0.0_real32) then
                        current = sample_logits(service_cache_logits, temperature, random_state, sampling_history, &
                            size(prompt_ids), options, adjusted_logits, history_counts, candidate_indices, &
                            candidate_values)
                    else if (sampling_penalties_active(options)) then
                        current = greedy_penalized(service_cache_logits, adjusted_logits, history_counts, &
                            sampling_history, size(prompt_ids), options)
                    else
                        current = int(maxloc(service_cache_logits, dim=1) - 1, int64)
                    end if
                else
                    current = service_cache_next_token
                end if
            end if
        end if
        do i = prompt_start, size(prompt_ids)
            if (temperature > 0.0_real32) then
                call service_model%forward(int(prompt_ids(i), int64), int(i - 1, int64), logits, stat, &
                    i == size(prompt_ids))
                if (stat%is_ok()) then
                    if (i == size(prompt_ids)) then
                        current = sample_logits(logits, temperature, random_state, sampling_history, i, options, &
                            adjusted_logits, history_counts, candidate_indices, candidate_values)
                    end if
                end if
            else
                if (sampling_penalties_active(options)) then
                    call service_model%forward(int(prompt_ids(i), int64), int(i - 1, int64), logits, stat, &
                        i == size(prompt_ids))
                    if (stat%is_ok()) then
                        if (i == size(prompt_ids)) then
                            current = greedy_penalized(logits, adjusted_logits, history_counts, sampling_history, i, &
                                options)
                        end if
                    end if
                else
                    call service_model%forward_greedy(int(prompt_ids(i), int64), int(i - 1, int64), &
                        next_token, logit_sum, stat)
                    if (stat%is_ok()) current = next_token
                end if
            end if
            if (.not. stat%is_ok()) then
                write(error_unit, '(a)') 'fortai-native: model forward failed: ' // trim(stat%message)
                service_cache_valid = .false.
                return
            end if
            state_count = i
            if (use_external_draft) then
                call service_draft_model%forward_greedy(int(prompt_ids(i), int64), int(i - 1, int64), &
                    draft_next_token, draft_logit_sum, draft_stat)
                if (.not. draft_stat%is_ok()) then
                    write(error_unit, '(a)') 'fortai-native: external draft forward failed: ' // &
                        trim(draft_stat%message)
                    service_cache_valid = .false.
                    return
                end if
            end if
        end do
        generated_count = 0
        do while (generated_count < max_tokens .and. size(prompt_ids) + generated_count < service_model%max_context)
            if (current < 0_int64) exit
            if (.not. ignore_eos .and. service_tokenizer%is_stop(int(current, int32))) exit
            generated_count = generated_count + 1
            generated_ids(generated_count) = int(current, int32)
            sampling_history(size(prompt_ids) + generated_count) = int(current, int32)
            if (generated_count == max_tokens) exit
            position = int(size(prompt_ids) + generated_count - 1, int64)
            state_step_speculative = .false.
            if (use_native_mtp .and. max_tokens - generated_count >= 2) then
                call service_model%forward_greedy_speculative(current, position, speculative_tokens, &
                    speculative_count, logit_sum, stat)
                if (.not. stat%is_ok()) then
                    write(error_unit, '(a)') 'fortai-native: speculative model forward failed: ' // trim(stat%message)
                    service_cache_valid = .false.
                    return
                end if
                if (speculative_count <= 0 .or. speculative_count > size(speculative_tokens)) then
                    write(error_unit, '(a)') 'fortai-native: speculative model forward returned no tokens'
                    service_cache_valid = .false.
                    return
                end if
                if (state_count + speculative_count > size(state_tokens)) then
                    write(error_unit, '(a)') 'fortai-native: speculative state exceeded cache workspace'
                    service_cache_valid = .false.
                    return
                end if
                state_tokens(state_count + 1) = int(current, int32)
                do j = 1, speculative_count - 1
                    state_tokens(state_count + j + 1) = int(speculative_tokens(j), int32)
                end do
                state_count = state_count + speculative_count
                state_step_speculative = .true.
                stop_generation = .false.
                ! Keep the final speculative token in CURRENT.  The loop's
                ! next iteration emits it before asking the model for more;
                ! this preserves the ordinary one-token state machine while
                ! allowing the preceding tokens to be emitted immediately.
                do j = 1, speculative_count - 1
                    if (.not. ignore_eos .and. service_tokenizer%is_stop(int(speculative_tokens(j), int32))) then
                        stop_generation = .true.
                        exit
                    end if
                    generated_count = generated_count + 1
                    generated_ids(generated_count) = int(speculative_tokens(j), int32)
                    sampling_history(size(prompt_ids) + generated_count) = int(speculative_tokens(j), int32)
                end do
                if (stop_generation) exit
                current = speculative_tokens(speculative_count)
            else if (temperature > 0.0_real32) then
                call service_model%forward(current, position, logits, stat)
                if (stat%is_ok()) current = sample_logits(logits, temperature, random_state, sampling_history, &
                    size(prompt_ids) + generated_count, options, adjusted_logits, history_counts, candidate_indices, &
                    candidate_values)
            else
                if (state_count >= size(state_tokens)) then
                    write(error_unit, '(a)') 'fortai-native: generation state exceeded cache workspace'
                    service_cache_valid = .false.
                    return
                end if
                state_tokens(state_count + 1) = int(current, int32)
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
                service_cache_valid = .false.
                return
            end if
            if (.not. state_step_speculative) state_count = state_count + 1
            if (use_external_draft) then
                call service_draft_model%forward_greedy(current, position, draft_next_token, draft_logit_sum, &
                    draft_stat)
                if (.not. draft_stat%is_ok()) then
                    write(error_unit, '(a)') 'fortai-native: external draft forward failed: ' // &
                        trim(draft_stat%message)
                    service_cache_valid = .false.
                    return
                end if
            end if
        end do
        if (generated_count > 0) then
            block
                character(len=:), allocatable :: decoded
                call service_tokenizer%decode(generated_ids(:generated_count), decoded)
                call output_text%set(decoded)
            end block
        end if
        if (state_count > 0) then
            allocate(new_cache_tokens(state_count))
            new_cache_tokens = state_tokens(:state_count)
            call move_alloc(new_cache_tokens, service_cache_tokens)
            service_cache_count = state_count
            service_cache_next_token = current
            preserve_old_logits = cache_hit .and. state_count == cache_count
            if (needs_logits) then
                if (allocated(logits)) then
                    allocate(new_cache_logits(size(logits)))
                    new_cache_logits = logits
                    call move_alloc(new_cache_logits, service_cache_logits)
                    service_cache_logits_valid = .true.
                else if (.not. preserve_old_logits) then
                    if (allocated(service_cache_logits)) deallocate(service_cache_logits)
                    service_cache_logits_valid = .false.
                end if
            else if (.not. preserve_old_logits) then
                if (allocated(service_cache_logits)) deallocate(service_cache_logits)
                service_cache_logits_valid = .false.
            end if
            service_cache_valid = .true.
        else
            service_cache_valid = .false.
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
