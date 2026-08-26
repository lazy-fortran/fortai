program fortai_server
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use, intrinsic :: iso_fortran_env, only: error_unit, output_unit, int32
    use fortai_native_service, only: fortai_native_service_close, fortai_native_service_init
    use fortai_status, only: status_t
    use fortai_whisper_service, only: fortai_whisper_service_close, fortai_whisper_service_init
    use fortai_string, only: string_t
    implicit none

    type :: server_config_t
        type(string_t) :: model
        type(string_t) :: host
        type(string_t) :: alias
        integer :: port = 8080
        integer :: context_size = 0
        integer :: threads = 0
        integer :: gpu_layers = 999
        integer :: main_gpu = 0
    end type server_config_t

    interface
        integer(c_int) function fortai_http_transport_run(host, port, model, cuda) &
                bind(C, name='fortai_http_transport_run')
            import c_char, c_int
            character(kind=c_char), intent(in) :: host(*), model(*)
            integer(c_int), value :: port, cuda
        end function fortai_http_transport_run

        integer(c_int) function fortai_server_set_environment(name, value) &
                bind(C, name='fortai_server_set_environment')
            import c_char, c_int
            character(kind=c_char), intent(in) :: name(*), value(*)
        end function fortai_server_set_environment

        integer(c_int) function fortai_server_online_cpus() bind(C, name='fortai_server_online_cpus')
            import c_int
        end function fortai_server_online_cpus
    end interface

    type(server_config_t) :: config
    logical :: okay, show_help, show_version, show_devices, show_cache_list, show_completion, whisper_mode
    integer(c_int) :: status, vocab, layers
    integer :: require_cuda
    character(kind=c_char), allocatable :: cmodel(:), chost(:)
    character(len=32) :: number
    type(status_t) :: whisper_status

    call parse_arguments(config, okay, show_help, show_version, show_devices, show_cache_list, show_completion)
    if (show_help) then
        call print_usage()
        stop
    end if
    if (show_version) then
        write(output_unit, '(a)') 'fortai-server 0.1 (FortAI-owned native Fortran runtime)'
        stop
    end if
    if (show_devices) then
        call print_devices()
        stop
    end if
    if (show_cache_list) then
        write(output_unit, '(a)') 'FortAI model cache: no native cache entries'
        stop
    end if
    if (show_completion) then
        write(output_unit, '(a)') '# FortAI accepts the llama.cpp server option surface.'
        write(output_unit, '(a)') '# Generate completion from the llama.cpp completion source if desired.'
        stop
    end if
    if (.not. okay) then
        call print_usage()
        error stop 2
    end if
    if (config%threads <= 0) config%threads = int(fortai_server_online_cpus())
    if (config%threads <= 0) config%threads = 1
    call configure_environment(config)
    whisper_mode = is_whisper_model_path(config%model%as_character())

    allocate(cmodel(config%model%length() + 1), chost(config%host%length() + 1))
    call config%model%to_c(cmodel, size(cmodel))
    call config%host%to_c(chost, size(chost))
    require_cuda = merge(1, 0, config%gpu_layers > 0)
    write(number, '(i0)') config%gpu_layers
    write(error_unit, '(a)') 'FORTAI_SERVER_BACKEND=fortai model=' // config%model%as_character() // &
        ' alias=' // config%alias%as_character() // &
        ' host=' // config%host%as_character() // ' port=' // trim(int_text(config%port)) // &
        ' threads=' // trim(int_text(config%threads)) // ' gpu_layers=' // trim(number)
    if (whisper_mode) then
        call fortai_whisper_service_init(config%model%as_character(), config%gpu_layers > 0, &
            int(config%main_gpu, int32), whisper_flash_enabled(), int(config%threads, int32), whisper_status)
        if (.not. whisper_status%is_ok()) then
            write(error_unit, '(a)') 'fortai-server: native Whisper initialization failed: ' // whisper_status%message
            error stop 1
        end if
        write(error_unit, '(a)') 'FORTAI_SERVER_READY=1 backend=fortai-whisper'
    else
        status = fortai_native_service_init(cmodel, int(config%context_size, c_int), int(config%threads, c_int), &
            int(config%gpu_layers, c_int), int(config%main_gpu, c_int), int(require_cuda, c_int), vocab, layers)
        if (status /= 0_c_int) then
            write(error_unit, '(a)') 'fortai-server: FortAI model context creation failed'
            error stop 1
        end if
        write(error_unit, '(a)') 'FORTAI_SERVER_READY=1 vocab=' // trim(int_text(int(vocab))) // &
            ' layers=' // trim(int_text(int(layers)))
    end if
    status = fortai_http_transport_run(chost, int(config%port, c_int), cmodel, &
        int(merge(1, 0, config%gpu_layers > 0), c_int))
    if (whisper_mode) then
        call fortai_whisper_service_close()
    else
        call fortai_native_service_close()
    end if
    if (status /= 0_c_int) error stop 1

contains

    function argument_at(index) result(argument)
        integer, intent(in) :: index
        type(string_t) :: argument
        character(len=4096) :: raw
        integer :: length
        raw = ''
        call get_command_argument(index, raw, length=length)
        if (length > len(raw)) then
            call argument%set('')
        else if (length > 0) then
            call argument%set(raw(:length))
        else
            call argument%set('')
        end if
    end function argument_at

    logical function parse_integer(argument, minimum, maximum, value)
        type(string_t), intent(in) :: argument
        integer, intent(in) :: minimum, maximum
        integer, intent(out) :: value
        character(len=:), allocatable :: text
        integer :: ios
        text = argument%as_character()
        value = 0
        read(text, *, iostat=ios) value
        parse_integer = ios == 0 .and. value >= minimum .and. value <= maximum
    end function parse_integer

    logical function parse_thread_count(argument, value)
        type(string_t), intent(in) :: argument
        integer, intent(out) :: value
        character(len=:), allocatable :: text
        integer :: ios

        text = argument%as_character()
        value = 0
        read(text, *, iostat=ios) value
        if (ios /= 0) then
            parse_thread_count = .false.
            return
        end if
        ! llama-server uses -1 for the automatic thread count.  FortAI's
        ! zero value is the same policy and is resolved against online CPUs
        ! immediately before native model initialization.
        if (value == -1) value = 0
        parse_thread_count = value >= 0 .and. value <= 4096
    end function parse_thread_count

    logical function parse_gpu_layers(argument, value)
        type(string_t), intent(in) :: argument
        integer, intent(out) :: value
        character(len=:), allocatable :: text
        integer :: ios

        text = argument%as_character()
        select case (trim(text))
        case ('auto', 'all')
            value = 999
            parse_gpu_layers = .true.
            return
        case ('none', 'off')
            value = 0
            parse_gpu_layers = .true.
            return
        end select
        value = 0
        read(text, *, iostat=ios) value
        parse_gpu_layers = ios == 0 .and. value >= 0 .and. value <= 8192
    end function parse_gpu_layers

    subroutine parse_arguments(config, okay, show_help, show_version, show_devices, show_cache_list, show_completion)
        type(server_config_t), intent(inout) :: config
        logical, intent(out) :: okay, show_help, show_version, show_devices, show_cache_list, show_completion
        integer :: i, count, value
        type(string_t) :: argument, option
        character(len=:), allocatable :: option_text, value_text
        character(len=4096) :: device
        character(len=4096) :: model_env
        character(len=256) :: alias_env
        integer :: device_length, model_length, alias_length

        okay = .true.; show_help = .false.; show_version = .false.
        show_devices = .false.; show_cache_list = .false.; show_completion = .false.
        call config%host%set('127.0.0.1')
        ! Normalize llama-server's canonical LLAMA_ARG_* environment names
        ! before reading model defaults or applying command-line overrides.
        call import_llama_environment()
        call get_environment_variable('FORTAI_SERVER_MODEL', model_env, length=model_length)
        if (model_length <= 0) then
            call get_environment_variable('LLAMACPP_MODEL', model_env, length=model_length)
        end if
        if (model_length > 0) then
            if (model_length <= len(model_env)) call config%model%set(model_env(:model_length))
        end if
        call get_environment_variable('FORTAI_SERVER_ALIAS', alias_env, length=alias_length)
        if (alias_length <= 0) call get_environment_variable('LLAMACPP_SERVED_ALIAS', alias_env, length=alias_length)
        if (alias_length <= 0) call get_environment_variable('LLAMACPP_MODEL_ALIAS', alias_env, length=alias_length)
        if (alias_length > 0) then
            if (alias_length <= len(alias_env)) call config%alias%set(alias_env(:alias_length))
        end if
        if (config%alias%length() == 0) call config%alias%set('qwen')
        call load_environment_defaults(config, okay)
        if (.not. okay) return
        count = command_argument_count()
        i = 1
        do while (i <= count)
            argument = argument_at(i)
            option = argument
            option_text = option%as_character()
            select case (option_text)
            case ('--help', '-h', '--usage')
                show_help = .true.; return
            case ('--version')
                show_version = .true.; return
            case ('--list-devices')
                show_devices = .true.; return
            case ('-cl', '--cache-list')
                show_cache_list = .true.; return
            case ('--completion-bash')
                show_completion = .true.; return
            case ('-m', '--model')
                i = i + 1; if (i > count) then; okay = .false.; return; end if
                call argument_text_at(i, value_text)
                call config%model%set(value_text)
            case ('--host')
                i = i + 1; if (i > count) then; okay = .false.; return; end if
                call argument_text_at(i, value_text)
                call config%host%set(value_text)
            case ('--port')
                i = i + 1
                if (i > count) then; okay = .false.; return; end if
                if (.not. parse_integer(argument_at(i), 1, 65535, value)) then; okay = .false.; return; end if
                config%port = value
            case ('-c', '--ctx-size')
                i = i + 1
                if (i > count) then; okay = .false.; return; end if
                if (.not. parse_integer(argument_at(i), 0, 2**20, value)) then; okay = .false.; return; end if
                config%context_size = value
            case ('-t', '--threads')
                i = i + 1
                if (i > count) then; okay = .false.; return; end if
                if (.not. parse_thread_count(argument_at(i), value)) then; okay = .false.; return; end if
                config%threads = value
            case ('-ngl', '--n-gpu-layers', '--gpu-layers')
                i = i + 1
                if (i > count) then; okay = .false.; return; end if
                if (.not. parse_gpu_layers(argument_at(i), value)) then; okay = .false.; return; end if
                config%gpu_layers = value
            case ('--main-gpu', '-mg')
                i = i + 1
                if (i > count) then; okay = .false.; return; end if
                if (.not. parse_integer(argument_at(i), 0, 255, value)) then; okay = .false.; return; end if
                config%main_gpu = value
            case ('--parallel', '-np', '--tensor-split', '-ts', '--split-mode', '-sm', '--model-draft', '--spec-type', &
                    '--spec-draft-model', '-md', '--spec-draft-n-max', '--flash-attn', '-fa', '--cache-type-k', '-ctk', &
                    '--cache-type-v', '-ctv', '--cache-type-k-draft', '--spec-draft-type-k', '-ctkd', &
                    '--cache-type-v-draft', '--spec-draft-type-v', '-ctvd', '--batch-size', '-b', '--ubatch-size', '-ub', &
                    '--fit', '-fit', '--cache-ram', '-cram', '--cache-reuse', &
                    '--n-cpu-moe', '--ncmoe', '-ncmoe', '--reasoning-budget', '--mmproj', '-mm', '--rpc', '--threads-http', &
                    '--load-mode', '-lm', '--mmproj-device', '-mmdev', &
                    '--temp', '--temperature', '--top-k', '--top-p', '--min-p', '--repeat-penalty', &
                    '--presence-penalty', '--frequency-penalty', '--repeat-last-n', '--seed', '-s', '--reasoning-format', &
                    '--reasoning', '-rea', '--reasoning-effort', '--threads-batch', '-tb', '--n-predict', '--predict', '-n', &
                    '--max-tokens', '--device', '-dev', '--chat-template-kwargs')
                i = i + 1; if (i > count) then; okay = .false.; return; end if
                call argument_text_at(i, value_text)
                call set_option_environment(option_text, value_text)
            case ('-C', '--cpu-mask', '-Cr', '--cpu-range', '--cpu-strict', '--prio', '--poll', &
                    '-Cb', '--cpu-mask-batch', '-Crb', '--cpu-range-batch', '--cpu-strict-batch', '--prio-batch', &
                    '--poll-batch', '--keep', '--rope-scaling', '--rope-scale', '--rope-freq-base', '--rope-freq-scale', &
                    '--yarn-orig-ctx', '--yarn-ext-factor', '--yarn-attn-factor', '--yarn-beta-slow', '--yarn-beta-fast', &
                    '-dt', '--defrag-thold', '--override-tensor', '-ot', '--fit-target', '-fitt', '--fit-ctx', '-fitc', &
                    '--numa', &
                    '--override-kv', '--lora', '--lora-scaled', '--control-vector', '--control-vector-scaled', &
                    '-mu', '--model-url', '-dr', '--docker-repo', '-hf', '-hfr', '--hf-repo', '-hff', '--hf-file', &
                    '-hfv', '-hfrv', '--hf-repo-v', '-hffv', '--hf-file-v', '-hft', '--hf-token', '--log-file', &
                    '--log-colors', '-lv', '--verbosity', '--log-verbosity', '--samplers', '--sampler-seq', '--sampling-seq', &
                    '--top-nsigma', '--top-n-sigma', '--xtc-probability', '--xtc-threshold', '--typical', '--typical-p', &
                    '--dry-multiplier', '--dry-base', '--dry-allowed-length', '--dry-penalty-last-n', &
                    '--dry-sequence-breaker', '--adaptive-target', '--adaptive-decay', '--dynatemp-range', '--dynatemp-exp', &
                    '--mirostat', '--mirostat-lr', '--mirostat-ent', '--logit-bias', '-l', '--grammar', '--grammar-file', &
                    '--json-schema', '-j', '--json-schema-file', '-jf', '-lcs', '--lookup-cache-static', '-lcd', &
                    '--lookup-cache-dynamic', '-ctxcp', '--ctx-checkpoints', '--swa-checkpoints', '-cms', &
                    '--checkpoint-min-step', '--reverse-prompt', '-r', '--pooling', '-mmu', '--mmproj-url', &
                    '--image-min-tokens', '--image-max-tokens', '--mtmd-batch-max-tokens', '--tags', '--embd-normalize', &
                    '--timeout', '-to', '--sse-ping-interval', '--slot-save-path', '--media-path', '--models-dir', &
                    '--models-preset', '--models-max', '--reasoning-budget-message', '--chat-template', &
                    '--chat-template-file', '--slot-prompt-similarity', '-sps', '--sleep-idle-seconds', '--api-key', &
                    '--api-key-file', '--ssl-key-file', '--ssl-cert-file', '--path', '--api-prefix', '--cors-origins', &
                    '--cors-methods', '--cors-headers', '--tools-runtime', '--mcp-servers-config', '--mcp-servers-json', &
                    '--ui-config', &
                    '--webui-config', '--ui-config-file', '--webui-config-file', '--tools', '--log-prompts-dir', &
                    '--spec-draft-hf', '-hfd', '-hfrd', '--hf-repo-draft', '--spec-draft-threads', '-td', &
                    '--threads-draft', '--spec-draft-threads-batch', '-tbd', '--threads-batch-draft', &
                    '--spec-draft-cpu-mask', '-Cd', '--cpu-mask-draft', '--spec-draft-cpu-range', '-Crd', &
                    '--cpu-range-draft', '--spec-draft-cpu-strict', '--cpu-strict-draft', '--spec-draft-prio', &
                    '--prio-draft', '--spec-draft-poll', '--poll-draft', '--spec-draft-cpu-mask-batch', '-Cbd', &
                    '--cpu-mask-batch-draft', '--spec-draft-cpu-strict-batch', '--cpu-strict-batch-draft', &
                    '--spec-draft-prio-batch', '--prio-batch-draft', '--spec-draft-poll-batch', '--poll-batch-draft', &
                    '--spec-draft-override-tensor', '-otd', '--override-tensor-draft', '--spec-draft-n-cpu-moe', &
                    '--spec-draft-ncmoe', '-ncmoed', '--n-cpu-moe-draft', &
                    '--spec-draft-n-min', '--spec-draft-p-split', '--draft-p-split', '--spec-draft-p-min', '--draft-p-min', &
                    '--spec-draft-device', '-devd', '--device-draft', '--spec-draft-ngl', '-ngld', '--gpu-layers-draft', &
                    '--n-gpu-layers-draft', '--spec-ngram-mod-n-min', '--spec-ngram-mod-n-max', '--spec-ngram-mod-n-match', &
                    '--spec-ngram-simple-size-n', '--spec-ngram-simple-size-m', '--spec-ngram-simple-min-hits', &
                    '--spec-ngram-map-k-size-n', '--spec-ngram-map-k-size-m', '--spec-ngram-map-k-min-hits', &
                    '--spec-ngram-map-k4v-size-n', '--spec-ngram-map-k4v-size-m', '--spec-ngram-map-k4v-min-hits', &
                    '--draft', '--draft-n', '--draft-max', '--draft-min', '--draft-n-min', '--spec-ngram-size-n', &
                    '--spec-ngram-size-m', '--spec-ngram-min-hits', '-mv', '--model-vocoder')
                i = i + 1; if (i > count) then; okay = .false.; return; end if
                call argument_text_at(i, value_text)
                call set_option_environment(option_text, value_text)
            case ('--control-vector-layer-range')
                i = i + 1; if (i > count) then; okay = .false.; return; end if
                call argument_text_at(i, value_text)
                i = i + 1; if (i > count) then; okay = .false.; return; end if
                call argument_text_at(i, option_text)
                call set_option_environment('--control-vector-layer-range', trim(value_text) // ',' // trim(option_text))
            case ('--alias', '-a', '--served-model-name')
                i = i + 1; if (i > count) then; okay = .false.; return; end if
                call argument_text_at(i, value_text)
                call config%alias%set(value_text)
            case ('--jinja', '--no-jinja', '--mmap', '--no-mmap', '--mlock', '--kv-offload', '--no-kv-offload', &
                    '--context-shift', '--no-context-shift', '--metrics', '--log-timestamps', '--mmproj-offload', &
                    '--no-mmproj-offload', '--webui', '--no-ui', '--no-webui', '--op-offload', '--no-op-offload', &
                    '--reasoning-preserve', '--no-reasoning-preserve', '--ui')
                call set_flag_environment(option_text)
            case ('--swa-full', '--perf', '--no-perf', '-e', '--escape', '--no-escape', '--repack', '-nr', '--no-repack', &
                    '--no-host', '--direct-io', '-dio', '--no-direct-io', '-ndio', '--check-tensors', '--cpu-moe', '-cmoe', &
                    '--log-disable', '-v', '--verbose', '--log-verbose', '--offline', '--log-prefix', '--no-log-prefix', &
                    '--no-log-timestamps', '-kvo', '-nkvo', '--spec-draft-backend-sampling', &
                    '--no-spec-draft-backend-sampling', '--tts-use-guide-tokens', &
                    '--spec-draft-cpu-moe', '-cmoed', '--cpu-moe-draft', &
                    '--embd-gemma-default', '--fim-qwen-1', '--fim-qwen-1.5b-default', '--fim-qwen-3b-default', &
                    '--fim-qwen-7b-default', &
                    '--fim-qwen-7b-spec', '--fim-qwen-14b-spec', '--fim-qwen-30b-default', '--gpt-oss-20b-default', &
                    '--gpt-oss-120b-default', '--vision-gemma-4b-default', '--vision-gemma-12b-default', '--spec-default', &
                    '-kvu', '--kv-unified', '-no-kvu', '--no-kv-unified', '--cache-idle-slots', '--no-cache-idle-slots', &
                    '--warmup', '--no-warmup', '-sp', '--special', '--spm-infill', '-cb', '--cont-batching', '-nocb', &
                    '--no-cont-batching', '--mmproj-auto', '--no-mmproj', '--no-mmproj-auto', '--ui-mcp-proxy', &
                    '--webui-mcp-proxy', '--no-ui-mcp-proxy', '--no-webui-mcp-proxy', '-ag', '--agent', '-no-ag', &
                    '--no-agent', '--embedding', '--embeddings', '--rerank', '--reranking', '--reuse-port', '--props', &
                    '--slots', '--no-slots', '--cache-prompt', '--no-cache-prompt', '--prefill-assistant', &
                    '--no-prefill-assistant', '--skip-chat-parsing', '--no-skip-chat-parsing', '--ignore-eos', &
                    '--cors-credentials', '--no-cors-credentials', &
                    '-bs', '--backend-sampling', '--lora-init-without-apply', '--models-autoload', '--no-models-autoload')
                call set_flag_environment(option_text)
            case default
                if (len(option_text) > 0) then
                    if (option_text(1:1) /= '-' .and. config%model%length() == 0) then
                        call config%model%set(option_text)
                    else
                        okay = .false.; return
                    end if
                else
                    okay = .false.; return
                end if
            end select
            i = i + 1
        end do
        device = ''; call get_environment_variable('FORTAI_SERVER_DEVICE', device, length=device_length)
        if (device_length > 0 .and. device_length <= len(device)) then
            select case (trim(device(:device_length)))
            case ('cpu', 'host', 'none')
                config%gpu_layers = 0
            end select
            call apply_device_selection(device(:device_length), config)
        end if
        if (config%model%length() == 0) okay = .false.
    end subroutine parse_arguments

    subroutine apply_device_selection(value, config)
        character(len=*), intent(in) :: value
        type(server_config_t), intent(inout) :: config
        character(len=:), allocatable :: first, second
        integer :: comma, first_device, second_device

        if (len_trim(value) == 0) return
        if (trim(value) == 'cpu' .or. trim(value) == 'host' .or. trim(value) == 'none') return
        comma = index(value, ',')
        if (comma > 1 .and. comma < len_trim(value)) then
            first = trim(value(:comma - 1))
            second = trim(value(comma + 1:))
        else
            first = trim(value)
            allocate(character(len=0) :: second)
        end if
        if (parse_device_index(first, first_device)) config%main_gpu = first_device
        if (comma > 0 .and. parse_device_index(second, second_device)) then
            call set_environment('FORTAI_CUDA_Q4_SECOND_DEVICE', int_text(second_device))
        else if (comma == 0) then
            ! A single explicit device is the llama.cpp equivalent of a
            ! one-board placement; do not silently split Q4 weights onto the
            ! next visible CUDA device.
            call set_environment('FORTAI_CUDA_Q4_SPLIT', '0')
        end if
    end subroutine apply_device_selection

    logical function parse_device_index(value, index_value)
        character(len=*), intent(in) :: value
        integer, intent(out) :: index_value
        character(len=:), allocatable :: digits
        integer :: i, first_digit, last_digit, ios

        parse_device_index = .false.
        index_value = 0
        first_digit = 0
        last_digit = 0
        do i = 1, len_trim(value)
            if (value(i:i) >= '0' .and. value(i:i) <= '9') then
                if (first_digit == 0) first_digit = i
                last_digit = i
            else if (first_digit > 0) then
                exit
            end if
        end do
        if (first_digit == 0 .or. last_digit < first_digit) return
        digits = value(first_digit:last_digit)
        read(digits, *, iostat=ios) index_value
        parse_device_index = ios == 0 .and. index_value >= 0 .and. index_value <= 255
    end function parse_device_index

    subroutine configure_environment(config)
        type(server_config_t), intent(in) :: config
        call set_environment('FORTAI_LLAMA_FASTPATH', 'native')
        call set_environment('OMP_NUM_THREADS', int_text(config%threads))
        call set_environment('FORTAI_THREADS', int_text(config%threads))
        call set_environment('FORTAI_CONTEXT', int_text(config%context_size))
        call set_environment('FORTAI_GPU_LAYERS', int_text(config%gpu_layers))
        call set_environment('FORTAI_MAIN_GPU', int_text(config%main_gpu))
        call set_environment('FORTAI_ENABLE_CUDA_GRAPH', '1')
        call set_environment('FORTAI_SERVER_ALIAS', config%alias%as_character())
    end subroutine configure_environment

    subroutine set_option_environment(option, value)
        character(len=*), intent(in) :: option, value
        character(len=64) :: name
        select case (option)
        case ('--parallel', '-np'); name = 'FORTAI_PARALLEL'
        case ('--tensor-split', '-ts'); name = 'FORTAI_TENSOR_SPLIT'
        case ('--split-mode', '-sm'); name = 'FORTAI_SPLIT_MODE'
        case ('--model-draft', '--spec-draft-model', '-md'); name = 'FORTAI_DRAFT_MODEL'
        case ('--spec-type'); name = 'FORTAI_SPEC_TYPE'
        case ('--spec-draft-n-max'); name = 'FORTAI_SPEC_DRAFT_N_MAX'
        case ('--flash-attn', '-fa'); name = 'FORTAI_FLASH_ATTN'
        case ('--cache-type-k', '-ctk'); name = 'FORTAI_CACHE_TYPE_K'
        case ('--cache-type-v', '-ctv'); name = 'FORTAI_CACHE_TYPE_V'
        case ('--cache-type-k-draft', '--spec-draft-type-k', '-ctkd'); name = 'FORTAI_CACHE_TYPE_K_DRAFT'
        case ('--cache-type-v-draft', '--spec-draft-type-v', '-ctvd'); name = 'FORTAI_CACHE_TYPE_V_DRAFT'
        case ('--ubatch-size', '-ub'); name = 'FORTAI_UBATCH'
        case ('--batch-size', '-b'); name = 'FORTAI_BATCH'
        case ('--cache-ram', '-cram'); name = 'FORTAI_CACHE_RAM'
        case ('--cache-reuse'); name = 'FORTAI_CACHE_REUSE'
        case ('--n-cpu-moe', '--ncmoe', '-ncmoe'); name = 'FORTAI_N_CPU_MOE'
        case ('--reasoning-budget'); name = 'FORTAI_REASONING_BUDGET'
        case ('--reasoning-effort'); name = 'FORTAI_REASONING_EFFORT'
        case ('--reasoning', '-rea'); name = 'FORTAI_REASONING'
        case ('--mmproj', '-mm'); name = 'FORTAI_MMPROJ'
        case ('--rpc'); name = 'FORTAI_RPC'
        case ('--threads-http'); name = 'FORTAI_THREADS_HTTP'
        case ('--load-mode', '-lm'); name = 'FORTAI_LOAD_MODE'
        case ('--mmproj-device', '-mmdev'); name = 'MTMD_BACKEND_DEVICE'
        case ('--fit', '-fit'); name = 'FORTAI_FIT'
        case ('--chat-template-kwargs'); name = 'FORTAI_CHAT_TEMPLATE_KWARGS'
        case ('--temp', '--temperature'); name = 'FORTAI_TEMPERATURE'
        case ('--top-k'); name = 'FORTAI_TOP_K'
        case ('--top-p'); name = 'FORTAI_TOP_P'
        case ('--min-p'); name = 'FORTAI_MIN_P'
        case ('--repeat-penalty'); name = 'FORTAI_REPEAT_PENALTY'
        case ('--presence-penalty'); name = 'FORTAI_PRESENCE_PENALTY'
        case ('--frequency-penalty'); name = 'FORTAI_FREQUENCY_PENALTY'
        case ('--repeat-last-n'); name = 'FORTAI_REPEAT_LAST_N'
        case ('--seed', '-s'); name = 'FORTAI_SEED'
        case ('--reasoning-format'); name = 'FORTAI_REASONING_FORMAT'
        case ('--threads-batch', '-tb'); name = 'FORTAI_THREADS_BATCH'
        case ('--n-predict', '--predict', '-n', '--max-tokens'); name = 'FORTAI_MAX_TOKENS'
        case ('--device', '-dev'); name = 'FORTAI_SERVER_DEVICE'
        case ('-C', '--cpu-mask'); name = 'FORTAI_CPU_MASK'
        case ('-Cr', '--cpu-range'); name = 'FORTAI_CPU_RANGE'
        case ('--cpu-strict'); name = 'FORTAI_CPU_STRICT'
        case ('--prio'); name = 'FORTAI_PRIO'
        case ('--poll'); name = 'FORTAI_POLL'
        case ('-Cb', '--cpu-mask-batch'); name = 'FORTAI_CPU_MASK_BATCH'
        case ('-Crb', '--cpu-range-batch'); name = 'FORTAI_CPU_RANGE_BATCH'
        case ('--cpu-strict-batch'); name = 'FORTAI_CPU_STRICT_BATCH'
        case ('--prio-batch'); name = 'FORTAI_PRIO_BATCH'
        case ('--poll-batch'); name = 'FORTAI_POLL_BATCH'
        case ('--keep'); name = 'FORTAI_KEEP'
        case ('--rope-scaling'); name = 'FORTAI_ROPE_SCALING_TYPE'
        case ('--rope-scale'); name = 'FORTAI_ROPE_SCALE'
        case ('--rope-freq-base'); name = 'FORTAI_ROPE_FREQ_BASE'
        case ('--rope-freq-scale'); name = 'FORTAI_ROPE_FREQ_SCALE'
        case ('--yarn-orig-ctx'); name = 'FORTAI_YARN_ORIG_CTX'
        case ('--yarn-ext-factor'); name = 'FORTAI_YARN_EXT_FACTOR'
        case ('--yarn-attn-factor'); name = 'FORTAI_YARN_ATTN_FACTOR'
        case ('--yarn-beta-slow'); name = 'FORTAI_YARN_BETA_SLOW'
        case ('--yarn-beta-fast'); name = 'FORTAI_YARN_BETA_FAST'
        case ('-dt', '--defrag-thold'); name = 'FORTAI_DEFRAG_THOLD'
        case ('--numa'); name = 'FORTAI_NUMA'
        case ('--override-tensor', '-ot'); name = 'FORTAI_OVERRIDE_TENSOR'
        case ('--fit-target', '-fitt'); name = 'FORTAI_FIT_TARGET'
        case ('--fit-ctx', '-fitc'); name = 'FORTAI_FIT_CTX'
        case ('--override-kv'); name = 'FORTAI_OVERRIDE_KV'
        case ('--lora'); name = 'FORTAI_LORA'
        case ('--lora-scaled'); name = 'FORTAI_LORA_SCALED'
        case ('--control-vector'); name = 'FORTAI_CONTROL_VECTOR'
        case ('--control-vector-scaled'); name = 'FORTAI_CONTROL_VECTOR_SCALED'
        case ('--control-vector-layer-range'); name = 'FORTAI_CONTROL_VECTOR_LAYER_RANGE'
        case ('-mu', '--model-url'); name = 'FORTAI_MODEL_URL'
        case ('-dr', '--docker-repo'); name = 'FORTAI_DOCKER_REPO'
        case ('-hf', '-hfr', '--hf-repo'); name = 'FORTAI_HF_REPO'
        case ('-hff', '--hf-file'); name = 'FORTAI_HF_FILE'
        case ('-hfv', '-hfrv', '--hf-repo-v'); name = 'FORTAI_HF_REPO_V'
        case ('-hffv', '--hf-file-v'); name = 'FORTAI_HF_FILE_V'
        case ('-hft', '--hf-token'); name = 'HF_TOKEN'
        case ('--log-file'); name = 'FORTAI_LOG_FILE'
        case ('--log-colors'); name = 'FORTAI_LOG_COLORS'
        case ('-lv', '--verbosity', '--log-verbosity'); name = 'FORTAI_LOG_VERBOSITY'
        case ('--samplers'); name = 'FORTAI_SAMPLERS'
        case ('--sampler-seq', '--sampling-seq'); name = 'FORTAI_SAMPLING_SEQ'
        case ('--top-nsigma', '--top-n-sigma'); name = 'FORTAI_TOP_N_SIGMA'
        case ('--xtc-probability'); name = 'FORTAI_XTC_PROBABILITY'
        case ('--xtc-threshold'); name = 'FORTAI_XTC_THRESHOLD'
        case ('--typical', '--typical-p'); name = 'FORTAI_TYPICAL_P'
        case ('--dry-multiplier'); name = 'FORTAI_DRY_MULTIPLIER'
        case ('--dry-base'); name = 'FORTAI_DRY_BASE'
        case ('--dry-allowed-length'); name = 'FORTAI_DRY_ALLOWED_LENGTH'
        case ('--dry-penalty-last-n'); name = 'FORTAI_DRY_PENALTY_LAST_N'
        case ('--dry-sequence-breaker'); name = 'FORTAI_DRY_SEQUENCE_BREAKER'
        case ('--adaptive-target'); name = 'FORTAI_ADAPTIVE_TARGET'
        case ('--adaptive-decay'); name = 'FORTAI_ADAPTIVE_DECAY'
        case ('--dynatemp-range'); name = 'FORTAI_DYNATEMP_RANGE'
        case ('--dynatemp-exp'); name = 'FORTAI_DYNATEMP_EXP'
        case ('--mirostat'); name = 'FORTAI_MIROSTAT'
        case ('--mirostat-lr'); name = 'FORTAI_MIROSTAT_LR'
        case ('--mirostat-ent'); name = 'FORTAI_MIROSTAT_ENT'
        case ('--logit-bias', '-l'); name = 'FORTAI_LOGIT_BIAS'
        case ('--grammar'); name = 'FORTAI_GRAMMAR'
        case ('--grammar-file'); name = 'FORTAI_GRAMMAR_FILE'
        case ('--json-schema', '-j'); name = 'FORTAI_JSON_SCHEMA'
        case ('--json-schema-file', '-jf'); name = 'FORTAI_JSON_SCHEMA_FILE'
        case ('-lcs', '--lookup-cache-static'); name = 'FORTAI_LOOKUP_CACHE_STATIC'
        case ('-lcd', '--lookup-cache-dynamic'); name = 'FORTAI_LOOKUP_CACHE_DYNAMIC'
        case ('-ctxcp', '--ctx-checkpoints', '--swa-checkpoints'); name = 'FORTAI_CTX_CHECKPOINTS'
        case ('-cms', '--checkpoint-min-step'); name = 'FORTAI_CHECKPOINT_MIN_SPACING_NT'
        case ('--reverse-prompt', '-r'); name = 'FORTAI_REVERSE_PROMPT'
        case ('--pooling'); name = 'FORTAI_POOLING'
        case ('-mmu', '--mmproj-url'); name = 'FORTAI_MMPROJ_URL'
        case ('--image-min-tokens'); name = 'FORTAI_IMAGE_MIN_TOKENS'
        case ('--image-max-tokens'); name = 'FORTAI_IMAGE_MAX_TOKENS'
        case ('--mtmd-batch-max-tokens'); name = 'FORTAI_MTMD_BATCH_MAX_TOKENS'
        case ('--tags'); name = 'FORTAI_TAGS'
        case ('--embd-normalize'); name = 'FORTAI_EMBD_NORMALIZE'
        case ('--timeout', '-to'); name = 'FORTAI_TIMEOUT'
        case ('--sse-ping-interval'); name = 'FORTAI_SSE_PING_INTERVAL'
        case ('--slot-save-path'); name = 'FORTAI_SLOT_SAVE_PATH'
        case ('--media-path'); name = 'FORTAI_MEDIA_PATH'
        case ('--models-dir'); name = 'FORTAI_MODELS_DIR'
        case ('--models-preset'); name = 'FORTAI_MODELS_PRESET'
        case ('--models-max'); name = 'FORTAI_MODELS_MAX'
        case ('--reasoning-budget-message'); name = 'FORTAI_REASONING_BUDGET_MESSAGE'
        case ('--chat-template'); name = 'FORTAI_CHAT_TEMPLATE'
        case ('--chat-template-file'); name = 'FORTAI_CHAT_TEMPLATE_FILE'
        case ('--slot-prompt-similarity', '-sps'); name = 'FORTAI_SLOT_PROMPT_SIMILARITY'
        case ('--sleep-idle-seconds'); name = 'FORTAI_SLEEP_IDLE_SECONDS'
        case ('--api-key'); name = 'LLAMA_API_KEY'
        case ('--api-key-file'); name = 'FORTAI_API_KEY_FILE'
        case ('--ssl-key-file'); name = 'FORTAI_SSL_KEY_FILE'
        case ('--ssl-cert-file'); name = 'FORTAI_SSL_CERT_FILE'
        case ('--path'); name = 'FORTAI_STATIC_PATH'
        case ('--api-prefix'); name = 'FORTAI_API_PREFIX'
        case ('--cors-origins'); name = 'FORTAI_CORS_ORIGINS'
        case ('--cors-methods'); name = 'FORTAI_CORS_METHODS'
        case ('--cors-headers'); name = 'FORTAI_CORS_HEADERS'
        case ('--tools-runtime'); name = 'FORTAI_TOOLS_RUNTIME'
        case ('--mcp-servers-config'); name = 'FORTAI_MCP_SERVERS_CONFIG'
        case ('--mcp-servers-json'); name = 'FORTAI_MCP_SERVERS_JSON'
        case ('--ui-config', '--webui-config'); name = 'FORTAI_UI_CONFIG'
        case ('--ui-config-file', '--webui-config-file'); name = 'FORTAI_UI_CONFIG_FILE'
        case ('--tools'); name = 'FORTAI_TOOLS'
        case ('--log-prompts-dir'); name = 'FORTAI_LOG_PROMPTS_DIR'
        case ('--spec-draft-hf', '-hfd', '-hfrd', '--hf-repo-draft'); name = 'FORTAI_SPEC_DRAFT_HF'
        case ('--spec-draft-threads', '-td', '--threads-draft'); name = 'FORTAI_SPEC_DRAFT_THREADS'
        case ('--spec-draft-threads-batch', '-tbd', '--threads-batch-draft'); name = 'FORTAI_SPEC_DRAFT_THREADS_BATCH'
        case ('--spec-draft-cpu-mask', '-Cd', '--cpu-mask-draft'); name = 'FORTAI_SPEC_DRAFT_CPU_MASK'
        case ('--spec-draft-cpu-range', '-Crd', '--cpu-range-draft'); name = 'FORTAI_SPEC_DRAFT_CPU_RANGE'
        case ('--spec-draft-cpu-strict', '--cpu-strict-draft'); name = 'FORTAI_SPEC_DRAFT_CPU_STRICT'
        case ('--spec-draft-prio', '--prio-draft'); name = 'FORTAI_SPEC_DRAFT_PRIO'
        case ('--spec-draft-poll', '--poll-draft'); name = 'FORTAI_SPEC_DRAFT_POLL'
        case ('--spec-draft-cpu-mask-batch', '-Cbd', '--cpu-mask-batch-draft'); name = 'FORTAI_SPEC_DRAFT_CPU_MASK_BATCH'
        case ('--spec-draft-cpu-strict-batch', '--cpu-strict-batch-draft'); name = 'FORTAI_SPEC_DRAFT_CPU_STRICT_BATCH'
        case ('--spec-draft-prio-batch', '--prio-batch-draft'); name = 'FORTAI_SPEC_DRAFT_PRIO_BATCH'
        case ('--spec-draft-poll-batch', '--poll-batch-draft'); name = 'FORTAI_SPEC_DRAFT_POLL_BATCH'
        case ('--spec-draft-override-tensor', '-otd', '--override-tensor-draft'); name = 'FORTAI_SPEC_DRAFT_OVERRIDE_TENSOR'
        case ('--spec-draft-n-cpu-moe', '--spec-draft-ncmoe', '-ncmoed', '--n-cpu-moe-draft'); name = 'FORTAI_SPEC_DRAFT_N_CPU_MOE'
        case ('--spec-draft-n-min'); name = 'FORTAI_SPEC_DRAFT_N_MIN'
        case ('--spec-draft-p-split', '--draft-p-split'); name = 'FORTAI_SPEC_DRAFT_P_SPLIT'
        case ('--spec-draft-p-min', '--draft-p-min'); name = 'FORTAI_SPEC_DRAFT_P_MIN'
        case ('--spec-draft-device', '-devd', '--device-draft'); name = 'FORTAI_SPEC_DRAFT_DEVICE'
        case ('--spec-draft-ngl', '-ngld', '--gpu-layers-draft', '--n-gpu-layers-draft'); name = 'FORTAI_SPEC_DRAFT_N_GPU_LAYERS'
        case ('--spec-ngram-mod-n-min'); name = 'FORTAI_SPEC_NGRAM_MOD_N_MIN'
        case ('--spec-ngram-mod-n-max'); name = 'FORTAI_SPEC_NGRAM_MOD_N_MAX'
        case ('--spec-ngram-mod-n-match'); name = 'FORTAI_SPEC_NGRAM_MOD_N_MATCH'
        case ('--spec-ngram-simple-size-n'); name = 'FORTAI_SPEC_NGRAM_SIMPLE_SIZE_N'
        case ('--spec-ngram-simple-size-m'); name = 'FORTAI_SPEC_NGRAM_SIMPLE_SIZE_M'
        case ('--spec-ngram-simple-min-hits'); name = 'FORTAI_SPEC_NGRAM_SIMPLE_MIN_HITS'
        case ('--spec-ngram-map-k-size-n'); name = 'FORTAI_SPEC_NGRAM_MAP_K_SIZE_N'
        case ('--spec-ngram-map-k-size-m'); name = 'FORTAI_SPEC_NGRAM_MAP_K_SIZE_M'
        case ('--spec-ngram-map-k-min-hits'); name = 'FORTAI_SPEC_NGRAM_MAP_K_MIN_HITS'
        case ('--spec-ngram-map-k4v-size-n'); name = 'FORTAI_SPEC_NGRAM_MAP_K4V_SIZE_N'
        case ('--spec-ngram-map-k4v-size-m'); name = 'FORTAI_SPEC_NGRAM_MAP_K4V_SIZE_M'
        case ('--spec-ngram-map-k4v-min-hits'); name = 'FORTAI_SPEC_NGRAM_MAP_K4V_MIN_HITS'
        case ('--draft', '--draft-n', '--draft-max', '--draft-min', '--draft-n-min'); name = 'FORTAI_DRAFT_N'
        case ('--spec-ngram-size-n'); name = 'FORTAI_SPEC_NGRAM_SIZE_N'
        case ('--spec-ngram-size-m'); name = 'FORTAI_SPEC_NGRAM_SIZE_M'
        case ('--spec-ngram-min-hits'); name = 'FORTAI_SPEC_NGRAM_MIN_HITS'
        case ('-mv', '--model-vocoder'); name = 'FORTAI_MODEL_VOCODER'
        case default; name = 'FORTAI_OPTION'
        end select
        call set_environment(trim(name), value)
    end subroutine set_option_environment

    subroutine set_flag_environment(option)
        character(len=*), intent(in) :: option
        character(len=64) :: name, value
        select case (option)
        case ('--jinja'); name = 'FORTAI_JINJA'; value = '1'
        case ('--no-jinja'); name = 'FORTAI_JINJA'; value = '0'
        case ('--mmap'); name = 'FORTAI_NO_MMAP'; value = '0'
        case ('--no-mmap'); name = 'FORTAI_NO_MMAP'; value = '1'
        case ('--mlock'); name = 'FORTAI_MLOCK'; value = '1'
        case ('--kv-offload'); name = 'FORTAI_NO_KV_OFFLOAD'; value = '0'
        case ('--no-kv-offload'); name = 'FORTAI_NO_KV_OFFLOAD'; value = '1'
        case ('--context-shift'); name = 'FORTAI_NO_CONTEXT_SHIFT'; value = '0'
        case ('--no-context-shift'); name = 'FORTAI_NO_CONTEXT_SHIFT'; value = '1'
        case ('--metrics'); name = 'FORTAI_METRICS'; value = '1'
        case ('--log-timestamps'); name = 'FORTAI_LOG_TIMESTAMPS'; value = '1'
        case ('--mmproj-offload'); name = 'FORTAI_MMPROJ_OFFLOAD'; value = '1'
        case ('--no-mmproj-offload'); name = 'FORTAI_MMPROJ_OFFLOAD'; value = '0'
        case ('--webui', '--ui'); name = 'FORTAI_NO_WEBUI'; value = '0'
        case ('--no-ui', '--no-webui'); name = 'FORTAI_NO_WEBUI'; value = '1'
        case ('--op-offload'); name = 'FORTAI_NO_OP_OFFLOAD'; value = '0'
        case ('--no-op-offload'); name = 'FORTAI_NO_OP_OFFLOAD'; value = '1'
        case ('--reasoning-preserve'); name = 'FORTAI_REASONING_PRESERVE'; value = '1'
        case ('--no-reasoning-preserve'); name = 'FORTAI_REASONING_PRESERVE'; value = '0'
        case ('--swa-full'); name = 'FORTAI_SWA_FULL'; value = '1'
        case ('--perf'); name = 'FORTAI_PERF'; value = '1'
        case ('--no-perf'); name = 'FORTAI_PERF'; value = '0'
        case ('-e', '--escape'); name = 'FORTAI_ESCAPE'; value = '1'
        case ('--no-escape'); name = 'FORTAI_ESCAPE'; value = '0'
        case ('--repack', '-nr'); name = 'FORTAI_REPACK'; value = '1'
        case ('--no-repack'); name = 'FORTAI_REPACK'; value = '0'
        case ('--no-host'); name = 'FORTAI_NO_HOST'; value = '1'
        case ('--direct-io', '-dio'); name = 'FORTAI_DIRECT_IO'; value = '1'
        case ('--no-direct-io', '-ndio'); name = 'FORTAI_DIRECT_IO'; value = '0'
        case ('--check-tensors'); name = 'FORTAI_CHECK_TENSORS'; value = '1'
        case ('--cpu-moe', '-cmoe'); name = 'FORTAI_CPU_MOE'; value = '1'
        case ('--log-disable'); name = 'FORTAI_LOG_DISABLE'; value = '1'
        case ('-v', '--verbose', '--log-verbose'); name = 'FORTAI_LOG_VERBOSITY'; value = '5'
        case ('--offline'); name = 'FORTAI_OFFLINE'; value = '1'
        case ('--log-prefix'); name = 'FORTAI_LOG_PREFIX'; value = '1'
        case ('--no-log-prefix'); name = 'FORTAI_LOG_PREFIX'; value = '0'
        case ('--no-log-timestamps'); name = 'FORTAI_LOG_TIMESTAMPS'; value = '0'
        case ('-kvo'); name = 'FORTAI_NO_KV_OFFLOAD'; value = '0'
        case ('-nkvo'); name = 'FORTAI_NO_KV_OFFLOAD'; value = '1'
        case ('-kvu', '--kv-unified'); name = 'FORTAI_KV_UNIFIED'; value = '1'
        case ('-no-kvu', '--no-kv-unified'); name = 'FORTAI_KV_UNIFIED'; value = '0'
        case ('--cache-idle-slots'); name = 'FORTAI_CACHE_IDLE_SLOTS'; value = '1'
        case ('--no-cache-idle-slots'); name = 'FORTAI_CACHE_IDLE_SLOTS'; value = '0'
        case ('--warmup'); name = 'FORTAI_WARMUP'; value = '1'
        case ('--no-warmup'); name = 'FORTAI_WARMUP'; value = '0'
        case ('-sp', '--special'); name = 'FORTAI_SPECIAL'; value = '1'
        case ('--spm-infill'); name = 'FORTAI_SPM_INFILL'; value = '1'
        case ('-cb', '--cont-batching'); name = 'FORTAI_CONT_BATCHING'; value = '1'
        case ('-nocb', '--no-cont-batching'); name = 'FORTAI_CONT_BATCHING'; value = '0'
        case ('--mmproj-auto'); name = 'FORTAI_MMPROJ_AUTO'; value = '1'
        case ('--no-mmproj', '--no-mmproj-auto'); name = 'FORTAI_MMPROJ_AUTO'; value = '0'
        case ('--ui-mcp-proxy', '--webui-mcp-proxy'); name = 'FORTAI_UI_MCP_PROXY'; value = '1'
        case ('--no-ui-mcp-proxy', '--no-webui-mcp-proxy'); name = 'FORTAI_UI_MCP_PROXY'; value = '0'
        case ('-ag', '--agent'); name = 'FORTAI_AGENT'; value = '1'
        case ('-no-ag', '--no-agent'); name = 'FORTAI_AGENT'; value = '0'
        case ('--embedding', '--embeddings'); name = 'FORTAI_EMBEDDINGS'; value = '1'
        case ('--rerank', '--reranking'); name = 'FORTAI_RERANKING'; value = '1'
        case ('--reuse-port'); name = 'FORTAI_REUSE_PORT'; value = '1'
        case ('--cors-credentials'); name = 'FORTAI_CORS_CREDENTIALS'; value = '1'
        case ('--no-cors-credentials'); name = 'FORTAI_CORS_CREDENTIALS'; value = '0'
        case ('--props'); name = 'FORTAI_ENDPOINT_PROPS'; value = '1'
        case ('--slots'); name = 'FORTAI_ENDPOINT_SLOTS'; value = '1'
        case ('--no-slots'); name = 'FORTAI_ENDPOINT_SLOTS'; value = '0'
        case ('--cache-prompt'); name = 'FORTAI_CACHE_PROMPT'; value = '1'
        case ('--no-cache-prompt'); name = 'FORTAI_CACHE_PROMPT'; value = '0'
        case ('--prefill-assistant'); name = 'FORTAI_PREFILL_ASSISTANT'; value = '1'
        case ('--no-prefill-assistant'); name = 'FORTAI_PREFILL_ASSISTANT'; value = '0'
        case ('--skip-chat-parsing'); name = 'FORTAI_SKIP_CHAT_PARSING'; value = '1'
        case ('--no-skip-chat-parsing'); name = 'FORTAI_SKIP_CHAT_PARSING'; value = '0'
        case ('--ignore-eos'); name = 'FORTAI_IGNORE_EOS'; value = '1'
        case ('--spec-draft-backend-sampling'); name = 'FORTAI_SPEC_DRAFT_BACKEND_SAMPLING'; value = '1'
        case ('--no-spec-draft-backend-sampling'); name = 'FORTAI_SPEC_DRAFT_BACKEND_SAMPLING'; value = '0'
        case ('--spec-draft-cpu-moe', '-cmoed', '--cpu-moe-draft'); name = 'FORTAI_SPEC_DRAFT_CPU_MOE'; value = '1'
        case ('--tts-use-guide-tokens'); name = 'FORTAI_TTS_USE_GUIDE_TOKENS'; value = '1'
        case ('--embd-gemma-default'); name = 'FORTAI_EMBD_GEMMA_DEFAULT'; value = '1'
        case ('--fim-qwen-1', '--fim-qwen-1.5b-default', '--fim-qwen-3b-default', '--fim-qwen-7b-default', '--fim-qwen-7b-spec', &
                '--fim-qwen-14b-spec', '--fim-qwen-30b-default', '--gpt-oss-20b-default', '--gpt-oss-120b-default', &
                '--vision-gemma-4b-default', '--vision-gemma-12b-default', '--spec-default')
            name = 'FORTAI_PRESET'; value = option
        case ('-bs', '--backend-sampling'); name = 'FORTAI_BACKEND_SAMPLING'; value = '1'
        case ('--lora-init-without-apply'); name = 'FORTAI_LORA_INIT_WITHOUT_APPLY'; value = '1'
        case ('--models-autoload'); name = 'FORTAI_MODELS_AUTOLOAD'; value = '1'
        case ('--no-models-autoload'); name = 'FORTAI_MODELS_AUTOLOAD'; value = '0'
        case default; return
        end select
        call set_environment(trim(name), trim(value))
    end subroutine set_flag_environment

    subroutine load_environment_defaults(config, okay)
        type(server_config_t), intent(inout) :: config
        logical, intent(out) :: okay
        okay = .true.
        call load_text_environment('FORTAI_SERVER_HOST', 'LLAMACPP_HOST', config%host)
        call load_integer_environment('FORTAI_SERVER_PORT', 'LLAMACPP_PORT', 1, 65535, config%port, okay)
        call load_integer_environment('FORTAI_CONTEXT', 'LLAMACPP_CONTEXT', 0, 2**20, config%context_size, okay)
        call load_integer_environment('FORTAI_THREADS', 'LLAMACPP_THREADS', 0, 4096, config%threads, okay)
        call load_integer_environment('FORTAI_GPU_LAYERS', 'LLAMACPP_GPU_LAYERS', 0, 8192, config%gpu_layers, okay)
        call load_integer_environment('FORTAI_MAIN_GPU', 'LLAMACPP_MAIN_GPU', 0, 255, config%main_gpu, okay)
        if (.not. okay) return
        call load_alias_environment(config%alias)
        call load_text_option('FORTAI_PARALLEL', 'LLAMACPP_PARALLEL')
        call load_text_option('FORTAI_TENSOR_SPLIT', 'LLAMACPP_TENSOR_SPLIT')
        call load_text_option('FORTAI_SPLIT_MODE', 'LLAMACPP_SPLIT_MODE')
        call load_text_option('FORTAI_DRAFT_MODEL', 'LLAMACPP_DRAFT_MODEL')
        call load_text_option('FORTAI_SPEC_TYPE', 'LLAMACPP_SPEC_TYPE')
        call load_text_option('FORTAI_SPEC_DRAFT_N_MAX', 'LLAMACPP_SPEC_DRAFT_N_MAX')
        call load_text_option('FORTAI_FLASH_ATTN', 'LLAMACPP_FLASH_ATTN')
        call load_text_option('FORTAI_CACHE_TYPE_K', 'LLAMACPP_CACHE_TYPE_K')
        call load_text_option('FORTAI_CACHE_TYPE_V', 'LLAMACPP_CACHE_TYPE_V')
        call load_text_option('FORTAI_CACHE_TYPE_K_DRAFT', 'LLAMACPP_CACHE_TYPE_K_DRAFT')
        call load_text_option('FORTAI_CACHE_TYPE_V_DRAFT', 'LLAMACPP_CACHE_TYPE_V_DRAFT')
        call load_text_option('FORTAI_BATCH', 'LLAMACPP_BATCH')
        call load_text_option('FORTAI_UBATCH', 'LLAMACPP_UBATCH')
        call load_text_option('FORTAI_FIT', 'LLAMACPP_FIT')
        call load_text_option('FORTAI_CACHE_RAM', 'LLAMACPP_CACHE_RAM')
        call load_text_option('FORTAI_CACHE_REUSE', 'LLAMACPP_CACHE_REUSE')
        call load_text_option('FORTAI_N_CPU_MOE', 'LLAMACPP_N_CPU_MOE')
        call load_text_option('FORTAI_REASONING_BUDGET', 'LLAMACPP_REASONING_BUDGET')
        call load_text_option('FORTAI_MMPROJ', 'LLAMACPP_MMPROJ')
        call load_text_option('FORTAI_MMPROJ_OFFLOAD', 'LLAMACPP_MMPROJ_OFFLOAD')
        call load_text_option('FORTAI_LOAD_MODE', 'LLAMACPP_LOAD_MODE')
        call load_text_option('MTMD_BACKEND_DEVICE', 'LLAMACPP_MMPROJ_DEVICE')
        call load_text_option('FORTAI_RPC', 'LLAMACPP_RPC')
        call load_text_option('FORTAI_THREADS_HTTP', 'LLAMACPP_THREADS_HTTP')
        call load_text_option('FORTAI_CHAT_TEMPLATE_KWARGS', 'LLAMACPP_CHAT_TEMPLATE_KWARGS')
        call load_text_option('FORTAI_TEMPERATURE', 'LLAMACPP_TEMPERATURE')
        call load_text_option('FORTAI_TOP_K', 'LLAMACPP_TOP_K')
        call load_text_option('FORTAI_TOP_P', 'LLAMACPP_TOP_P')
        call load_text_option('FORTAI_MIN_P', 'LLAMACPP_MIN_P')
        call load_text_option('FORTAI_REPEAT_PENALTY', 'LLAMACPP_REPEAT_PENALTY')
        call load_text_option('FORTAI_PRESENCE_PENALTY', 'LLAMACPP_PRESENCE_PENALTY')
        call load_text_option('FORTAI_FREQUENCY_PENALTY', 'LLAMACPP_FREQUENCY_PENALTY')
        call load_text_option('FORTAI_REPEAT_LAST_N', 'LLAMACPP_REPEAT_LAST_N')
        call load_text_option('FORTAI_SEED', 'LLAMACPP_SEED')
        call load_text_option('FORTAI_REASONING_FORMAT', 'LLAMACPP_REASONING_FORMAT')
        call load_text_option('FORTAI_MAX_TOKENS', 'LLAMACPP_MAX_TOKENS')
        call load_text_option('FORTAI_SERVER_DEVICE', 'LLAMACPP_DEVICE')
        call load_text_option('FORTAI_NO_CONTEXT_SHIFT', 'LLAMACPP_NO_CONTEXT_SHIFT')
        call load_text_option('FORTAI_METRICS', 'LLAMACPP_METRICS')
        call load_text_option('FORTAI_LOG_TIMESTAMPS', 'LLAMACPP_LOG_TIMESTAMPS')
        call load_text_option('FORTAI_NO_MMAP', 'LLAMACPP_NO_MMAP')
        call load_text_option('FORTAI_MLOCK', 'LLAMACPP_MLOCK')
        call load_text_option('FORTAI_NO_KV_OFFLOAD', 'LLAMACPP_NO_KV_OFFLOAD')
        call load_text_option('FORTAI_NO_WEBUI', 'LLAMACPP_NO_WEBUI')
        call load_text_option('FORTAI_NO_OP_OFFLOAD', 'LLAMACPP_NO_OP_OFFLOAD')
        call load_text_option('FORTAI_CPU_MASK', 'LLAMACPP_CPU_MASK')
        call load_text_option('FORTAI_CPU_RANGE', 'LLAMACPP_CPU_RANGE')
        call load_text_option('FORTAI_CPU_STRICT', 'LLAMACPP_CPU_STRICT')
        call load_text_option('FORTAI_PRIO', 'LLAMACPP_PRIO')
        call load_text_option('FORTAI_POLL', 'LLAMACPP_POLL')
        call load_text_option('FORTAI_CPU_MASK_BATCH', 'LLAMACPP_CPU_MASK_BATCH')
        call load_text_option('FORTAI_CPU_RANGE_BATCH', 'LLAMACPP_CPU_RANGE_BATCH')
        call load_text_option('FORTAI_CPU_STRICT_BATCH', 'LLAMACPP_CPU_STRICT_BATCH')
        call load_text_option('FORTAI_PRIO_BATCH', 'LLAMACPP_PRIO_BATCH')
        call load_text_option('FORTAI_POLL_BATCH', 'LLAMACPP_POLL_BATCH')
        call load_text_option('FORTAI_KEEP', 'LLAMACPP_KEEP')
        call load_text_option('FORTAI_SWA_FULL', 'LLAMACPP_SWA_FULL')
        call load_text_option('FORTAI_PERF', 'LLAMACPP_PERF')
        call load_text_option('FORTAI_ESCAPE', 'LLAMACPP_ESCAPE')
        call load_text_option('FORTAI_ROPE_SCALING_TYPE', 'LLAMACPP_ROPE_SCALING_TYPE')
        call load_text_option('FORTAI_ROPE_SCALE', 'LLAMACPP_ROPE_SCALE')
        call load_text_option('FORTAI_ROPE_FREQ_BASE', 'LLAMACPP_ROPE_FREQ_BASE')
        call load_text_option('FORTAI_ROPE_FREQ_SCALE', 'LLAMACPP_ROPE_FREQ_SCALE')
        call load_text_option('FORTAI_YARN_ORIG_CTX', 'LLAMACPP_YARN_ORIG_CTX')
        call load_text_option('FORTAI_YARN_EXT_FACTOR', 'LLAMACPP_YARN_EXT_FACTOR')
        call load_text_option('FORTAI_YARN_ATTN_FACTOR', 'LLAMACPP_YARN_ATTN_FACTOR')
        call load_text_option('FORTAI_YARN_BETA_SLOW', 'LLAMACPP_YARN_BETA_SLOW')
        call load_text_option('FORTAI_YARN_BETA_FAST', 'LLAMACPP_YARN_BETA_FAST')
        call load_text_option('FORTAI_DEFRAG_THOLD', 'LLAMACPP_DEFRAG_THOLD')
        call load_text_option('FORTAI_DIRECT_IO', 'LLAMACPP_DIO')
        call load_text_option('FORTAI_NUMA', 'LLAMACPP_NUMA')
        call load_text_option('FORTAI_OVERRIDE_TENSOR', 'LLAMACPP_OVERRIDE_TENSOR')
        call load_text_option('FORTAI_CPU_MOE', 'LLAMACPP_CPU_MOE')
        call load_text_option('FORTAI_NO_HOST', 'LLAMACPP_NO_HOST')
        call load_text_option('FORTAI_REPACK', 'LLAMACPP_REPACK')
        call load_text_option('FORTAI_FIT_TARGET', 'LLAMACPP_FIT_TARGET')
        call load_text_option('FORTAI_FIT_CTX', 'LLAMACPP_FIT_CTX')
        call load_text_option('FORTAI_OVERRIDE_KV', 'LLAMACPP_OVERRIDE_KV')
        call load_text_option('FORTAI_MODEL_URL', 'LLAMACPP_MODEL_URL')
        call load_text_option('FORTAI_DOCKER_REPO', 'LLAMACPP_DOCKER_REPO')
        call load_text_option('FORTAI_HF_REPO', 'LLAMACPP_HF_REPO')
        call load_text_option('FORTAI_HF_FILE', 'LLAMACPP_HF_FILE')
        call load_text_option('FORTAI_HF_REPO_V', 'LLAMACPP_HF_REPO_V')
        call load_text_option('FORTAI_HF_FILE_V', 'LLAMACPP_HF_FILE_V')
        call load_text_option('FORTAI_LOG_FILE', 'LLAMACPP_LOG_FILE')
        call load_text_option('FORTAI_LOG_COLORS', 'LLAMACPP_LOG_COLORS')
        call load_text_option('FORTAI_LOG_VERBOSITY', 'LLAMACPP_LOG_VERBOSITY')
        call load_text_option('FORTAI_SAMPLERS', 'LLAMACPP_SAMPLERS')
        call load_text_option('FORTAI_SAMPLING_SEQ', 'LLAMACPP_SAMPLING_SEQ')
        call load_text_option('FORTAI_TOP_N_SIGMA', 'LLAMACPP_TOP_N_SIGMA')
        call load_text_option('FORTAI_XTC_PROBABILITY', 'LLAMACPP_XTC_PROBABILITY')
        call load_text_option('FORTAI_XTC_THRESHOLD', 'LLAMACPP_XTC_THRESHOLD')
        call load_text_option('FORTAI_TYPICAL_P', 'LLAMACPP_TYPICAL_P')
        call load_text_option('FORTAI_DRY_MULTIPLIER', 'LLAMACPP_DRY_MULTIPLIER')
        call load_text_option('FORTAI_DRY_BASE', 'LLAMACPP_DRY_BASE')
        call load_text_option('FORTAI_DRY_ALLOWED_LENGTH', 'LLAMACPP_DRY_ALLOWED_LENGTH')
        call load_text_option('FORTAI_DRY_PENALTY_LAST_N', 'LLAMACPP_DRY_PENALTY_LAST_N')
        call load_text_option('FORTAI_DRY_SEQUENCE_BREAKER', 'LLAMACPP_DRY_SEQUENCE_BREAKER')
        call load_text_option('FORTAI_ADAPTIVE_TARGET', 'LLAMACPP_ADAPTIVE_TARGET')
        call load_text_option('FORTAI_ADAPTIVE_DECAY', 'LLAMACPP_ADAPTIVE_DECAY')
        call load_text_option('FORTAI_DYNATEMP_RANGE', 'LLAMACPP_DYNATEMP_RANGE')
        call load_text_option('FORTAI_DYNATEMP_EXP', 'LLAMACPP_DYNATEMP_EXP')
        call load_text_option('FORTAI_MIROSTAT', 'LLAMACPP_MIROSTAT')
        call load_text_option('FORTAI_MIROSTAT_LR', 'LLAMACPP_MIROSTAT_LR')
        call load_text_option('FORTAI_MIROSTAT_ENT', 'LLAMACPP_MIROSTAT_ENT')
        call load_text_option('FORTAI_LOGIT_BIAS', 'LLAMACPP_LOGIT_BIAS')
        call load_text_option('FORTAI_GRAMMAR', 'LLAMACPP_GRAMMAR')
        call load_text_option('FORTAI_GRAMMAR_FILE', 'LLAMACPP_GRAMMAR_FILE')
        call load_text_option('FORTAI_JSON_SCHEMA', 'LLAMACPP_JSON_SCHEMA')
        call load_text_option('FORTAI_JSON_SCHEMA_FILE', 'LLAMACPP_JSON_SCHEMA_FILE')
        call load_text_option('FORTAI_POOLING', 'LLAMACPP_POOLING')
        call load_text_option('FORTAI_TIMEOUT', 'LLAMACPP_TIMEOUT')
        call load_text_option('FORTAI_THREADS_HTTP', 'LLAMACPP_THREADS_HTTP')
        call load_text_option('FORTAI_SSE_PING_INTERVAL', 'LLAMACPP_SSE_PING_INTERVAL')
        call load_text_option('FORTAI_CACHE_PROMPT', 'LLAMACPP_CACHE_PROMPT')
        call load_text_option('FORTAI_KV_UNIFIED', 'LLAMACPP_KV_UNIFIED')
        call load_text_option('FORTAI_CONT_BATCHING', 'LLAMACPP_CONT_BATCHING')
        call load_text_option('FORTAI_REUSE_PORT', 'LLAMACPP_REUSE_PORT')
        call load_text_option('FORTAI_STATIC_PATH', 'LLAMACPP_STATIC_PATH')
        call load_text_option('FORTAI_API_PREFIX', 'LLAMACPP_API_PREFIX')
        call load_text_option('FORTAI_CORS_ORIGINS', 'LLAMACPP_CORS_ORIGINS')
        call load_text_option('FORTAI_CORS_METHODS', 'LLAMACPP_CORS_METHODS')
        call load_text_option('FORTAI_CORS_HEADERS', 'LLAMACPP_CORS_HEADERS')
        call load_text_option('FORTAI_CORS_CREDENTIALS', 'LLAMACPP_CORS_CREDENTIALS')
        call load_text_option('FORTAI_TOOLS_RUNTIME', 'LLAMACPP_TOOLS_RUNTIME')
        call load_text_option('FORTAI_MCP_SERVERS_CONFIG', 'LLAMACPP_MCP_SERVERS_CONFIG')
        call load_text_option('FORTAI_MCP_SERVERS_JSON', 'LLAMACPP_MCP_SERVERS_JSON')
        call load_text_option('FORTAI_UI_CONFIG', 'LLAMACPP_UI_CONFIG')
        call load_text_option('FORTAI_UI_CONFIG_FILE', 'LLAMACPP_UI_CONFIG_FILE')
        call load_text_option('FORTAI_ENDPOINT_PROPS', 'LLAMACPP_ENDPOINT_PROPS')
        call load_text_option('FORTAI_ENDPOINT_SLOTS', 'LLAMACPP_ENDPOINT_SLOTS')
        call load_text_option('FORTAI_EMBEDDINGS', 'LLAMACPP_EMBEDDINGS')
        call load_text_option('FORTAI_RERANKING', 'LLAMACPP_RERANKING')
        call load_text_option('FORTAI_CHAT_TEMPLATE', 'LLAMACPP_CHAT_TEMPLATE')
        call load_text_option('FORTAI_CHAT_TEMPLATE_FILE', 'LLAMACPP_CHAT_TEMPLATE_FILE')
        call load_text_option('FORTAI_SKIP_CHAT_PARSING', 'LLAMACPP_SKIP_CHAT_PARSING')
        call load_text_option('FORTAI_PREFILL_ASSISTANT', 'LLAMACPP_PREFILL_ASSISTANT')
    end subroutine load_environment_defaults

    subroutine load_alias_environment(alias)
        type(string_t), intent(inout) :: alias
        character(len=4096) :: value
        integer :: length

        ! Keep the same precedence as the canonical llama arguments: an
        ! explicitly supplied LLAMA_ARG_ALIAS (normalized to
        ! FORTAI_SERVER_ALIAS) wins over the legacy LLAMACPP_* names.
        value = ''
        call get_environment_variable('FORTAI_SERVER_ALIAS', value, length=length)
        if (length <= 0) call get_environment_variable('FORTAI_ALIAS', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_SERVED_ALIAS', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_MODEL_ALIAS', value, length=length)
        if (length > 0 .and. length <= len(value)) call alias%set(value(:length))
    end subroutine load_alias_environment

    subroutine import_llama_environment()
        call import_environment_alias('FORTAI_SERVER_MODEL', 'LLAMA_ARG_MODEL')
        call import_environment_alias('FORTAI_SERVER_HOST', 'LLAMA_ARG_HOST')
        call import_environment_alias('FORTAI_SERVER_PORT', 'LLAMA_ARG_PORT')
        call import_environment_alias('FORTAI_CONTEXT', 'LLAMA_ARG_CTX_SIZE')
        call import_threads_alias()
        call import_gpu_layers_alias()
        call import_environment_alias('FORTAI_MAIN_GPU', 'LLAMA_ARG_MAIN_GPU')
        call import_environment_alias('FORTAI_SERVER_ALIAS', 'LLAMA_ARG_ALIAS')
        call import_environment_alias('FORTAI_PARALLEL', 'LLAMA_ARG_N_PARALLEL')
        call import_environment_alias('FORTAI_TENSOR_SPLIT', 'LLAMA_ARG_TENSOR_SPLIT')
        call import_environment_alias('FORTAI_SPLIT_MODE', 'LLAMA_ARG_SPLIT_MODE')
        call import_environment_alias('FORTAI_DRAFT_MODEL', 'LLAMA_ARG_SPEC_DRAFT_MODEL')
        call import_environment_alias('FORTAI_DRAFT_MODEL', 'LLAMA_ARG_MODEL_DRAFT')
        call import_environment_alias('FORTAI_SPEC_TYPE', 'LLAMA_ARG_SPEC_TYPE')
        call import_environment_alias('FORTAI_SPEC_DRAFT_N_MAX', 'LLAMA_ARG_SPEC_DRAFT_N_MAX')
        call import_environment_alias('FORTAI_FLASH_ATTN', 'LLAMA_ARG_FLASH_ATTN')
        call import_environment_alias('FORTAI_CACHE_TYPE_K', 'LLAMA_ARG_CACHE_TYPE_K')
        call import_environment_alias('FORTAI_CACHE_TYPE_V', 'LLAMA_ARG_CACHE_TYPE_V')
        call import_environment_alias('FORTAI_CACHE_TYPE_K_DRAFT', 'LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K')
        call import_environment_alias('FORTAI_CACHE_TYPE_V_DRAFT', 'LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V')
        call import_environment_alias('FORTAI_BATCH', 'LLAMA_ARG_BATCH')
        call import_environment_alias('FORTAI_UBATCH', 'LLAMA_ARG_UBATCH')
        call import_environment_alias('FORTAI_FIT', 'LLAMA_ARG_FIT')
        call import_environment_alias('FORTAI_CACHE_RAM', 'LLAMA_ARG_CACHE_RAM')
        call import_environment_alias('FORTAI_CACHE_REUSE', 'LLAMA_ARG_CACHE_REUSE')
        call import_environment_alias('FORTAI_N_CPU_MOE', 'LLAMA_ARG_N_CPU_MOE')
        call import_environment_alias('FORTAI_REASONING_BUDGET', 'LLAMA_ARG_THINK_BUDGET')
        call import_environment_alias('FORTAI_REASONING_EFFORT', 'LLAMA_ARG_REASONING_EFFORT')
        call import_environment_alias('FORTAI_REASONING', 'LLAMA_ARG_REASONING')
        call import_environment_alias('FORTAI_REASONING_PRESERVE', 'LLAMA_ARG_REASONING_PRESERVE')
        call import_environment_alias('FORTAI_MMPROJ', 'LLAMA_ARG_MMPROJ')
        call import_environment_alias('FORTAI_MMPROJ_OFFLOAD', 'LLAMA_ARG_MMPROJ_OFFLOAD')
        call import_environment_alias('FORTAI_LOAD_MODE', 'LLAMA_ARG_LOAD_MODE')
        call import_environment_alias('MTMD_BACKEND_DEVICE', 'LLAMA_ARG_MMPROJ_DEVICE')
        call import_environment_alias('FORTAI_RPC', 'LLAMA_ARG_RPC')
        call import_environment_alias('FORTAI_THREADS_HTTP', 'LLAMA_ARG_THREADS_HTTP')
        call import_environment_alias('FORTAI_CHAT_TEMPLATE_KWARGS', 'LLAMA_ARG_CHAT_TEMPLATE_KWARGS')
        call import_environment_alias('FORTAI_TEMPERATURE', 'LLAMA_ARG_TEMP')
        call import_environment_alias('FORTAI_TOP_K', 'LLAMA_ARG_TOP_K')
        call import_environment_alias('FORTAI_TOP_P', 'LLAMA_ARG_TOP_P')
        call import_environment_alias('FORTAI_MIN_P', 'LLAMA_ARG_MIN_P')
        call import_environment_alias('FORTAI_REPEAT_PENALTY', 'LLAMA_ARG_REPEAT_PENALTY')
        call import_environment_alias('FORTAI_PRESENCE_PENALTY', 'LLAMA_ARG_PRESENCE_PENALTY')
        call import_environment_alias('FORTAI_FREQUENCY_PENALTY', 'LLAMA_ARG_FREQUENCY_PENALTY')
        call import_environment_alias('FORTAI_REPEAT_LAST_N', 'LLAMA_ARG_REPEAT_LAST_N')
        call import_environment_alias('FORTAI_SEED', 'LLAMA_ARG_SEED')
        call import_environment_alias('FORTAI_REASONING_FORMAT', 'LLAMA_ARG_THINK')
        call import_environment_alias('FORTAI_MAX_TOKENS', 'LLAMA_ARG_N_PREDICT')
        call import_environment_alias('FORTAI_SERVER_DEVICE', 'LLAMA_ARG_DEVICE')
        call import_environment_alias('FORTAI_THREADS_BATCH', 'LLAMA_ARG_THREADS_BATCH')
        call import_environment_alias('FORTAI_MLOCK', 'LLAMA_ARG_MLOCK')
        call import_environment_alias('FORTAI_METRICS', 'LLAMA_ARG_ENDPOINT_METRICS')
        call import_environment_alias('FORTAI_LOG_TIMESTAMPS', 'LLAMA_ARG_LOG_TIMESTAMPS')
        call import_inverse_boolean_alias('FORTAI_NO_MMAP', 'LLAMA_ARG_MMAP')
        call import_inverse_boolean_alias('FORTAI_NO_KV_OFFLOAD', 'LLAMA_ARG_KV_OFFLOAD')
        call import_inverse_boolean_alias('FORTAI_NO_OP_OFFLOAD', 'LLAMA_ARG_OP_OFFLOAD')
        call import_inverse_boolean_alias('FORTAI_NO_CONTEXT_SHIFT', 'LLAMA_ARG_CONTEXT_SHIFT')
        call import_environment_alias('FORTAI_CPU_MASK', 'LLAMA_ARG_CPU_MASK')
        call import_environment_alias('FORTAI_CPU_RANGE', 'LLAMA_ARG_CPU_RANGE')
        call import_environment_alias('FORTAI_CPU_STRICT', 'LLAMA_ARG_CPU_STRICT')
        call import_environment_alias('FORTAI_PRIO', 'LLAMA_ARG_PRIO')
        call import_environment_alias('FORTAI_POLL', 'LLAMA_ARG_POLL')
        call import_environment_alias('FORTAI_CPU_MASK_BATCH', 'LLAMA_ARG_CPU_MASK_BATCH')
        call import_environment_alias('FORTAI_CPU_RANGE_BATCH', 'LLAMA_ARG_CPU_RANGE_BATCH')
        call import_environment_alias('FORTAI_CPU_STRICT_BATCH', 'LLAMA_ARG_CPU_STRICT_BATCH')
        call import_environment_alias('FORTAI_PRIO_BATCH', 'LLAMA_ARG_PRIO_BATCH')
        call import_environment_alias('FORTAI_POLL_BATCH', 'LLAMA_ARG_POLL_BATCH')
        call import_environment_alias('FORTAI_KEEP', 'LLAMA_ARG_KEEP')
        call import_environment_alias('FORTAI_SWA_FULL', 'LLAMA_ARG_SWA_FULL')
        call import_environment_alias('FORTAI_PERF', 'LLAMA_ARG_PERF')
        call import_environment_alias('FORTAI_ESCAPE', 'LLAMA_ARG_ESCAPE')
        call import_environment_alias('FORTAI_ROPE_SCALING_TYPE', 'LLAMA_ARG_ROPE_SCALING_TYPE')
        call import_environment_alias('FORTAI_ROPE_SCALE', 'LLAMA_ARG_ROPE_SCALE')
        call import_environment_alias('FORTAI_ROPE_FREQ_BASE', 'LLAMA_ARG_ROPE_FREQ_BASE')
        call import_environment_alias('FORTAI_ROPE_FREQ_SCALE', 'LLAMA_ARG_ROPE_FREQ_SCALE')
        call import_environment_alias('FORTAI_YARN_ORIG_CTX', 'LLAMA_ARG_YARN_ORIG_CTX')
        call import_environment_alias('FORTAI_YARN_EXT_FACTOR', 'LLAMA_ARG_YARN_EXT_FACTOR')
        call import_environment_alias('FORTAI_YARN_ATTN_FACTOR', 'LLAMA_ARG_YARN_ATTN_FACTOR')
        call import_environment_alias('FORTAI_YARN_BETA_SLOW', 'LLAMA_ARG_YARN_BETA_SLOW')
        call import_environment_alias('FORTAI_YARN_BETA_FAST', 'LLAMA_ARG_YARN_BETA_FAST')
        call import_environment_alias('FORTAI_DEFRAG_THOLD', 'LLAMA_ARG_DEFRAG_THOLD')
        call import_environment_alias('FORTAI_DIRECT_IO', 'LLAMA_ARG_DIO')
        call import_environment_alias('FORTAI_NUMA', 'LLAMA_ARG_NUMA')
        call import_environment_alias('FORTAI_OVERRIDE_TENSOR', 'LLAMA_ARG_OVERRIDE_TENSOR')
        call import_environment_alias('FORTAI_CPU_MOE', 'LLAMA_ARG_CPU_MOE')
        call import_environment_alias('FORTAI_NO_HOST', 'LLAMA_ARG_NO_HOST')
        call import_environment_alias('FORTAI_REPACK', 'LLAMA_ARG_REPACK')
        call import_environment_alias('FORTAI_FIT_TARGET', 'LLAMA_ARG_FIT_TARGET')
        call import_environment_alias('FORTAI_FIT_CTX', 'LLAMA_ARG_FIT_CTX')
        call import_environment_alias('FORTAI_OVERRIDE_KV', 'LLAMA_ARG_OVERRIDE_KV')
        call import_environment_alias('FORTAI_MODEL_URL', 'LLAMA_ARG_MODEL_URL')
        call import_environment_alias('FORTAI_DOCKER_REPO', 'LLAMA_ARG_DOCKER_REPO')
        call import_environment_alias('FORTAI_HF_REPO', 'LLAMA_ARG_HF_REPO')
        call import_environment_alias('FORTAI_HF_FILE', 'LLAMA_ARG_HF_FILE')
        call import_environment_alias('FORTAI_HF_REPO_V', 'LLAMA_ARG_HF_REPO_V')
        call import_environment_alias('FORTAI_HF_FILE_V', 'LLAMA_ARG_HF_FILE_V')
        call import_environment_alias('FORTAI_LOG_FILE', 'LLAMA_ARG_LOG_FILE')
        call import_environment_alias('FORTAI_LOG_COLORS', 'LLAMA_ARG_LOG_COLORS')
        call import_environment_alias('FORTAI_LOG_VERBOSITY', 'LLAMA_ARG_LOG_VERBOSITY')
        call import_environment_alias('FORTAI_SAMPLERS', 'LLAMA_ARG_SAMPLERS')
        call import_environment_alias('FORTAI_SAMPLING_SEQ', 'LLAMA_ARG_SAMPLING_SEQ')
        call import_environment_alias('FORTAI_TOP_N_SIGMA', 'LLAMA_ARG_TOP_N_SIGMA')
        call import_environment_alias('FORTAI_XTC_PROBABILITY', 'LLAMA_ARG_XTC_PROBABILITY')
        call import_environment_alias('FORTAI_XTC_THRESHOLD', 'LLAMA_ARG_XTC_THRESHOLD')
        call import_environment_alias('FORTAI_TYPICAL_P', 'LLAMA_ARG_TYPICAL_P')
        call import_environment_alias('FORTAI_DRY_MULTIPLIER', 'LLAMA_ARG_DRY_MULTIPLIER')
        call import_environment_alias('FORTAI_DRY_BASE', 'LLAMA_ARG_DRY_BASE')
        call import_environment_alias('FORTAI_DRY_ALLOWED_LENGTH', 'LLAMA_ARG_DRY_ALLOWED_LENGTH')
        call import_environment_alias('FORTAI_DRY_PENALTY_LAST_N', 'LLAMA_ARG_DRY_PENALTY_LAST_N')
        call import_environment_alias('FORTAI_DRY_SEQUENCE_BREAKER', 'LLAMA_ARG_DRY_SEQUENCE_BREAKER')
        call import_environment_alias('FORTAI_ADAPTIVE_TARGET', 'LLAMA_ARG_ADAPTIVE_TARGET')
        call import_environment_alias('FORTAI_ADAPTIVE_DECAY', 'LLAMA_ARG_ADAPTIVE_DECAY')
        call import_environment_alias('FORTAI_DYNATEMP_RANGE', 'LLAMA_ARG_DYNATEMP_RANGE')
        call import_environment_alias('FORTAI_DYNATEMP_EXP', 'LLAMA_ARG_DYNATEMP_EXP')
        call import_environment_alias('FORTAI_MIROSTAT', 'LLAMA_ARG_MIROSTAT')
        call import_environment_alias('FORTAI_MIROSTAT_LR', 'LLAMA_ARG_MIROSTAT_LR')
        call import_environment_alias('FORTAI_MIROSTAT_ENT', 'LLAMA_ARG_MIROSTAT_ENT')
        call import_environment_alias('FORTAI_LOGIT_BIAS', 'LLAMA_ARG_LOGIT_BIAS')
        call import_environment_alias('FORTAI_GRAMMAR', 'LLAMA_ARG_GRAMMAR')
        call import_environment_alias('FORTAI_GRAMMAR_FILE', 'LLAMA_ARG_GRAMMAR_FILE')
        call import_environment_alias('FORTAI_JSON_SCHEMA', 'LLAMA_ARG_JSON_SCHEMA')
        call import_environment_alias('FORTAI_JSON_SCHEMA_FILE', 'LLAMA_ARG_JSON_SCHEMA_FILE')
        call import_environment_alias('FORTAI_POOLING', 'LLAMA_ARG_POOLING')
        call import_environment_alias('FORTAI_TIMEOUT', 'LLAMA_ARG_TIMEOUT')
        call import_environment_alias('FORTAI_SSE_PING_INTERVAL', 'LLAMA_ARG_SSE_PING_INTERVAL')
        call import_environment_alias('FORTAI_CACHE_PROMPT', 'LLAMA_ARG_CACHE_PROMPT')
        call import_environment_alias('FORTAI_KV_UNIFIED', 'LLAMA_ARG_KV_UNIFIED')
        call import_environment_alias('FORTAI_CONT_BATCHING', 'LLAMA_ARG_CONT_BATCHING')
        call import_environment_alias('FORTAI_REUSE_PORT', 'LLAMA_ARG_REUSE_PORT')
        call import_environment_alias('FORTAI_STATIC_PATH', 'LLAMA_ARG_STATIC_PATH')
        call import_environment_alias('FORTAI_CORS_ORIGINS', 'LLAMA_ARG_CORS_ORIGINS')
        call import_environment_alias('FORTAI_CORS_METHODS', 'LLAMA_ARG_CORS_METHODS')
        call import_environment_alias('FORTAI_CORS_HEADERS', 'LLAMA_ARG_CORS_HEADERS')
        call import_environment_alias('FORTAI_CORS_CREDENTIALS', 'LLAMA_ARG_CORS_CREDENTIALS')
        call import_environment_alias('FORTAI_API_PREFIX', 'LLAMA_ARG_API_PREFIX')
        call import_environment_alias('FORTAI_UI_CONFIG', 'LLAMA_ARG_UI_CONFIG')
        call import_environment_alias('FORTAI_UI_CONFIG_FILE', 'LLAMA_ARG_UI_CONFIG_FILE')
        call import_environment_alias('FORTAI_ENDPOINT_PROPS', 'LLAMA_ARG_ENDPOINT_PROPS')
        call import_environment_alias('FORTAI_ENDPOINT_SLOTS', 'LLAMA_ARG_ENDPOINT_SLOTS')
        call import_environment_alias('FORTAI_EMBEDDINGS', 'LLAMA_ARG_EMBEDDINGS')
        call import_environment_alias('FORTAI_RERANKING', 'LLAMA_ARG_RERANKING')
        call import_environment_alias('FORTAI_CHAT_TEMPLATE', 'LLAMA_ARG_CHAT_TEMPLATE')
        call import_environment_alias('FORTAI_CHAT_TEMPLATE_FILE', 'LLAMA_ARG_CHAT_TEMPLATE_FILE')
        call import_environment_alias('FORTAI_SKIP_CHAT_PARSING', 'LLAMA_ARG_SKIP_CHAT_PARSING')
        call import_environment_alias('FORTAI_PREFILL_ASSISTANT', 'LLAMA_ARG_PREFILL_ASSISTANT')
        call import_environment_alias('FORTAI_OFFLINE', 'LLAMA_ARG_OFFLINE')
        call import_environment_alias('FORTAI_LOG_PREFIX', 'LLAMA_ARG_LOG_PREFIX')
        call import_environment_alias('FORTAI_BACKEND_SAMPLING', 'LLAMA_ARG_BACKEND_SAMPLING')
        call import_environment_alias('FORTAI_CTX_CHECKPOINTS', 'LLAMA_ARG_CTX_CHECKPOINTS')
        call import_environment_alias('FORTAI_CHECKPOINT_MIN_SPACING_NT', 'LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT')
        call import_environment_alias('FORTAI_CACHE_IDLE_SLOTS', 'LLAMA_ARG_CACHE_IDLE_SLOTS')
        call import_environment_alias('FORTAI_MMPROJ_URL', 'LLAMA_ARG_MMPROJ_URL')
        call import_environment_alias('FORTAI_MMPROJ_AUTO', 'LLAMA_ARG_MMPROJ_AUTO')
        call import_environment_alias('FORTAI_IMAGE_MIN_TOKENS', 'LLAMA_ARG_IMAGE_MIN_TOKENS')
        call import_environment_alias('FORTAI_IMAGE_MAX_TOKENS', 'LLAMA_ARG_IMAGE_MAX_TOKENS')
        call import_environment_alias('FORTAI_MTMD_BATCH_MAX_TOKENS', 'LLAMA_ARG_MTMD_BATCH_MAX_TOKENS')
        call import_environment_alias('FORTAI_TAGS', 'LLAMA_ARG_TAGS')
        call import_environment_alias('FORTAI_UI_MCP_PROXY', 'LLAMA_ARG_UI_MCP_PROXY')
        call import_environment_alias('FORTAI_TOOLS', 'LLAMA_ARG_TOOLS')
        call import_environment_alias('FORTAI_TOOLS_RUNTIME', 'LLAMA_ARG_TOOLS_RUNTIME')
        call import_environment_alias('FORTAI_MCP_SERVERS_CONFIG', 'LLAMA_ARG_MCP_SERVERS_CONFIG')
        call import_environment_alias('FORTAI_MCP_SERVERS_JSON', 'LLAMA_ARG_MCP_SERVERS_JSON')
        call import_environment_alias('FORTAI_AGENT', 'LLAMA_ARG_AGENT')
        call import_environment_alias('FORTAI_API_KEY_FILE', 'LLAMA_ARG_API_KEY_FILE')
        call import_environment_alias('FORTAI_SSL_KEY_FILE', 'LLAMA_ARG_SSL_KEY_FILE')
        call import_environment_alias('FORTAI_SSL_CERT_FILE', 'LLAMA_ARG_SSL_CERT_FILE')
        call import_environment_alias('FORTAI_MODELS_DIR', 'LLAMA_ARG_MODELS_DIR')
        call import_environment_alias('FORTAI_MODELS_PRESET', 'LLAMA_ARG_MODELS_PRESET')
        call import_environment_alias('FORTAI_MODELS_MAX', 'LLAMA_ARG_MODELS_MAX')
        call import_environment_alias('FORTAI_MODELS_AUTOLOAD', 'LLAMA_ARG_MODELS_AUTOLOAD')
        call import_environment_alias('FORTAI_JINJA', 'LLAMA_ARG_JINJA')
        call import_environment_alias('FORTAI_REASONING_BUDGET_MESSAGE', 'LLAMA_ARG_THINK_BUDGET_MESSAGE')
        call import_environment_alias('FORTAI_SPEC_DRAFT_N_GPU_LAYERS', 'LLAMA_ARG_N_GPU_LAYERS_DRAFT')
        call import_environment_alias('FORTAI_DRAFT_MAX', 'LLAMA_ARG_DRAFT_MAX')
        call import_environment_alias('FORTAI_DRAFT_MIN', 'LLAMA_ARG_DRAFT_MIN')
        call import_environment_alias('FORTAI_NO_MMAP', 'LLAMA_ARG_NO_MMAP')
        call import_environment_alias('FORTAI_LOG_PROMPTS_DIR', 'LLAMA_ARG_LOG_PROMPTS_DIR')
        call import_environment_alias('FORTAI_SPEC_DRAFT_HF', 'LLAMA_ARG_SPEC_DRAFT_HF_REPO')
        call import_environment_alias('FORTAI_SPEC_DRAFT_THREADS', 'LLAMA_ARG_SPEC_DRAFT_THREADS')
        call import_environment_alias('FORTAI_SPEC_DRAFT_THREADS_BATCH', 'LLAMA_ARG_SPEC_DRAFT_THREADS_BATCH')
        call import_environment_alias('FORTAI_SPEC_DRAFT_CPU_MASK', 'LLAMA_ARG_SPEC_DRAFT_CPU_MASK')
        call import_environment_alias('FORTAI_SPEC_DRAFT_CPU_RANGE', 'LLAMA_ARG_SPEC_DRAFT_CPU_RANGE')
        call import_environment_alias('FORTAI_SPEC_DRAFT_CPU_STRICT', 'LLAMA_ARG_SPEC_DRAFT_CPU_STRICT')
        call import_environment_alias('FORTAI_SPEC_DRAFT_PRIO', 'LLAMA_ARG_SPEC_DRAFT_PRIO')
        call import_environment_alias('FORTAI_SPEC_DRAFT_POLL', 'LLAMA_ARG_SPEC_DRAFT_POLL')
        call import_environment_alias('FORTAI_SPEC_DRAFT_CPU_MASK_BATCH', 'LLAMA_ARG_SPEC_DRAFT_CPU_MASK_BATCH')
        call import_environment_alias('FORTAI_SPEC_DRAFT_CPU_STRICT_BATCH', 'LLAMA_ARG_SPEC_DRAFT_CPU_STRICT_BATCH')
        call import_environment_alias('FORTAI_SPEC_DRAFT_PRIO_BATCH', 'LLAMA_ARG_SPEC_DRAFT_PRIO_BATCH')
        call import_environment_alias('FORTAI_SPEC_DRAFT_POLL_BATCH', 'LLAMA_ARG_SPEC_DRAFT_POLL_BATCH')
        call import_environment_alias('FORTAI_SPEC_DRAFT_N_CPU_MOE', 'LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE')
        call import_environment_alias('FORTAI_SPEC_DRAFT_CPU_MOE', 'LLAMA_ARG_SPEC_DRAFT_CPU_MOE')
        call import_environment_alias('FORTAI_SPEC_DRAFT_N_MIN', 'LLAMA_ARG_SPEC_DRAFT_N_MIN')
        call import_environment_alias('FORTAI_SPEC_DRAFT_P_SPLIT', 'LLAMA_ARG_SPEC_DRAFT_P_SPLIT')
        call import_environment_alias('FORTAI_SPEC_DRAFT_P_MIN', 'LLAMA_ARG_SPEC_DRAFT_P_MIN')
        call import_environment_alias('FORTAI_SPEC_DRAFT_BACKEND_SAMPLING', 'LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING')
        call import_environment_alias('FORTAI_SPEC_DRAFT_DEVICE', 'LLAMA_ARG_SPEC_DRAFT_DEVICE')
        call import_environment_alias('FORTAI_SPEC_DRAFT_N_GPU_LAYERS', 'LLAMA_ARG_SPEC_DRAFT_N_GPU_LAYERS')
        call import_environment_alias('FORTAI_SPEC_NGRAM_MOD_N_MIN', 'LLAMA_ARG_SPEC_NGRAM_MOD_N_MIN')
        call import_environment_alias('FORTAI_SPEC_NGRAM_MOD_N_MAX', 'LLAMA_ARG_SPEC_NGRAM_MOD_N_MAX')
        call import_environment_alias('FORTAI_SPEC_NGRAM_MOD_N_MATCH', 'LLAMA_ARG_SPEC_NGRAM_MOD_N_MATCH')
        call import_environment_alias('FORTAI_SPEC_NGRAM_SIMPLE_SIZE_N', 'LLAMA_ARG_SPEC_NGRAM_SIMPLE_SIZE_N')
        call import_environment_alias('FORTAI_SPEC_NGRAM_SIMPLE_SIZE_M', 'LLAMA_ARG_SPEC_NGRAM_SIMPLE_SIZE_M')
        call import_environment_alias('FORTAI_SPEC_NGRAM_SIMPLE_MIN_HITS', 'LLAMA_ARG_SPEC_NGRAM_SIMPLE_MIN_HITS')
        call import_environment_alias('FORTAI_SPEC_NGRAM_MAP_K_SIZE_N', 'LLAMA_ARG_SPEC_NGRAM_MAP_K_SIZE_N')
        call import_environment_alias('FORTAI_SPEC_NGRAM_MAP_K_SIZE_M', 'LLAMA_ARG_SPEC_NGRAM_MAP_K_SIZE_M')
        call import_environment_alias('FORTAI_SPEC_NGRAM_MAP_K_MIN_HITS', 'LLAMA_ARG_SPEC_NGRAM_MAP_K_MIN_HITS')
        call import_environment_alias('FORTAI_SPEC_NGRAM_MAP_K4V_SIZE_N', 'LLAMA_ARG_SPEC_NGRAM_MAP_K4V_SIZE_N')
        call import_environment_alias('FORTAI_SPEC_NGRAM_MAP_K4V_SIZE_M', 'LLAMA_ARG_SPEC_NGRAM_MAP_K4V_SIZE_M')
        call import_environment_alias('FORTAI_SPEC_NGRAM_MAP_K4V_MIN_HITS', 'LLAMA_ARG_SPEC_NGRAM_MAP_K4V_MIN_HITS')
        call import_environment_alias('FORTAI_MODEL_VOCODER', 'LLAMA_ARG_MODEL_VOCODER')
        call import_ui_alias()
    end subroutine import_llama_environment

    subroutine import_environment_alias(target_name, source_name)
        character(len=*), intent(in) :: target_name, source_name
        character(len=4096) :: value
        character(len=:), allocatable :: legacy_name
        integer :: length

        value = ''
        call get_environment_variable(target_name, value, length=length)
        if (length > 0) return
        value = ''
        call get_environment_variable(source_name, value, length=length)
        if (length <= 0) then
            if (index(target_name, 'FORTAI_') == 1) then
                if (len_trim(target_name) > 7) then
                    legacy_name = 'LLAMACPP_' // target_name(8:)
                    call get_environment_variable(legacy_name, value, length=length)
                end if
            end if
        end if
        if (length <= 0) then
            if (index(source_name, 'LLAMA_ARG_') == 1) then
                if (len_trim(source_name) > 10) then
                    legacy_name = 'LLAMACPP_' // source_name(11:)
                    call get_environment_variable(legacy_name, value, length=length)
                end if
            end if
        end if
        if (length <= 0) return
        if (length > len(value)) return
        call set_environment(target_name, value(:length))
    end subroutine import_environment_alias

    subroutine import_threads_alias()
        character(len=64) :: value
        integer :: length, ios, parsed

        value = ''
        call get_environment_variable('FORTAI_THREADS', value, length=length)
        if (length > 0) return
        value = ''
        call get_environment_variable('LLAMA_ARG_THREADS', value, length=length)
        if (length <= 0) return
        if (length > len(value)) return
        parsed = 0
        read(value(:length), *, iostat=ios) parsed
        if (ios /= 0) return
        if (parsed < 0) parsed = 0
        call set_environment('FORTAI_THREADS', int_text(parsed))
    end subroutine import_threads_alias

    subroutine import_gpu_layers_alias()
        character(len=64) :: value
        integer :: length, ios, parsed

        value = ''
        call get_environment_variable('FORTAI_GPU_LAYERS', value, length=length)
        if (length > 0) return
        value = ''
        call get_environment_variable('LLAMA_ARG_N_GPU_LAYERS', value, length=length)
        if (length <= 0) return
        if (length > len(value)) return
        select case (trim(value(:length)))
        case ('auto', 'all')
            call set_environment('FORTAI_GPU_LAYERS', '999')
            return
        case ('none', 'off')
            call set_environment('FORTAI_GPU_LAYERS', '0')
            return
        end select
        parsed = 0
        read(value(:length), *, iostat=ios) parsed
        if (ios == 0 .and. parsed >= 0) call set_environment('FORTAI_GPU_LAYERS', int_text(parsed))
    end subroutine import_gpu_layers_alias

    subroutine import_inverse_boolean_alias(target_name, source_name)
        character(len=*), intent(in) :: target_name, source_name
        character(len=64) :: value
        integer :: length

        value = ''
        call get_environment_variable(target_name, value, length=length)
        if (length > 0) return
        value = ''
        call get_environment_variable(source_name, value, length=length)
        if (length <= 0) return
        if (length > len(value)) return
        select case (trim(value(:length)))
        case ('0', 'false', 'off', 'no')
            call set_environment(target_name, '1')
        case default
            call set_environment(target_name, '0')
        end select
    end subroutine import_inverse_boolean_alias

    subroutine import_ui_alias()
        character(len=64) :: value
        integer :: length

        value = ''
        call get_environment_variable('FORTAI_NO_WEBUI', value, length=length)
        if (length > 0) return
        value = ''
        call get_environment_variable('LLAMA_ARG_UI', value, length=length)
        if (length <= 0) return
        if (length > len(value)) return
        select case (trim(value(:length)))
        case ('0', 'false', 'off', 'no')
            call set_environment('FORTAI_NO_WEBUI', '1')
        case default
            call set_environment('FORTAI_NO_WEBUI', '0')
        end select
    end subroutine import_ui_alias

    subroutine load_text_environment(primary, secondary, target)
        character(len=*), intent(in) :: primary, secondary
        type(string_t), intent(inout) :: target
        character(len=4096) :: text
        integer :: text_length
        text = ''; call get_environment_variable(primary, text, length=text_length)
        if (text_length <= 0) call get_environment_variable(secondary, text, length=text_length)
        if (text_length > 0 .and. text_length <= len(text)) call target%set(text(:text_length))
    end subroutine load_text_environment

    subroutine load_integer_environment(primary, secondary, minimum, maximum, target, valid)
        character(len=*), intent(in) :: primary, secondary
        integer, intent(in) :: minimum, maximum
        integer, intent(inout) :: target
        logical, intent(inout) :: valid
        character(len=64) :: text
        integer :: text_length, ios, candidate
        text = ''; call get_environment_variable(primary, text, length=text_length)
        if (text_length <= 0) call get_environment_variable(secondary, text, length=text_length)
        if (text_length <= 0) return
        if (text_length > len(text)) then; valid = .false.; return; end if
        candidate = 0
        read(text(:text_length), *, iostat=ios) candidate
        if (ios /= 0 .or. candidate < minimum .or. candidate > maximum) then
            valid = .false.
        else
            target = candidate
        end if
    end subroutine load_integer_environment

    subroutine load_text_option(primary, secondary)
        character(len=*), intent(in) :: primary, secondary
        character(len=4096) :: text
        integer :: text_length
        text = ''; call get_environment_variable(primary, text, length=text_length)
        if (text_length <= 0) call get_environment_variable(secondary, text, length=text_length)
        if (text_length > 0 .and. text_length <= len(text)) call set_environment(trim(primary), text(:text_length))
    end subroutine load_text_option

    subroutine set_environment(name, value)
        character(len=*), intent(in) :: name, value
        character(kind=c_char), allocatable :: cname(:), cvalue(:)
        integer(c_int) :: status
        allocate(cname(len_trim(name) + 1), cvalue(len_trim(value) + 1))
        call c_string(name, cname); call c_string(value, cvalue)
        status = fortai_server_set_environment(cname, cvalue)
    end subroutine set_environment

    subroutine c_string(text, output)
        character(len=*), intent(in) :: text
        character(kind=c_char), intent(out) :: output(:)
        type(string_t) :: value
        call value%set(trim(text))
        call value%to_c(output, size(output))
    end subroutine c_string

    function int_text(value) result(text)
        integer, intent(in) :: value
        character(len=32) :: text
        write(text, '(i0)') value
    end function int_text

    logical function is_whisper_model_path(path)
        character(len=*), intent(in) :: path
        character(len=:), allocatable :: lowered
        integer :: i, code, n, suffix_length

        lowered = trim(path)
        do i = 1, len(lowered)
            code = iachar(lowered(i:i))
            if (code >= iachar('A') .and. code <= iachar('Z')) lowered(i:i) = achar(code + 32)
        end do
        is_whisper_model_path = .false.
        do n = 1, 2
            if (n == 1) then
                suffix_length = 4
                if (len(lowered) >= suffix_length) then
                    if (lowered(len(lowered)-suffix_length+1:) == '.bin') is_whisper_model_path = .true.
                end if
            else
                suffix_length = 5
                if (len(lowered) >= suffix_length) then
                    if (lowered(len(lowered)-suffix_length+1:) == '.ggml') is_whisper_model_path = .true.
                end if
            end if
        end do
    end function is_whisper_model_path

    logical function whisper_flash_enabled()
        character(len=64) :: raw
        integer :: length, i, code

        raw = ''
        call get_environment_variable('FORTAI_FLASH_ATTN', raw, length=length)
        if (length <= 0) then
            call get_environment_variable('LLAMACPP_FLASH_ATTN', raw, length=length)
        end if
        whisper_flash_enabled = .true.
        if (length <= 0 .or. length > len(raw)) return
        do i = 1, length
            code = iachar(raw(i:i))
            if (code >= iachar('A') .and. code <= iachar('Z')) raw(i:i) = achar(code + 32)
        end do
        select case (trim(raw(:length)))
        case ('0', 'off', 'false', 'no', 'disabled')
            whisper_flash_enabled = .false.
        end select
    end function whisper_flash_enabled

    subroutine argument_text_at(index, text)
        integer, intent(in) :: index
        character(len=:), allocatable, intent(out) :: text
        type(string_t) :: argument
        argument = argument_at(index)
        text = argument%as_character()
    end subroutine argument_text_at

    subroutine print_devices()
        integer :: exit_status

        ! Keep this query side-effect free and use the same CUDA visibility
        ! rules as llama.cpp.  nvidia-smi is optional on CPU-only installs.
        call execute_command_line('nvidia-smi -L', wait=.true., exitstat=exit_status)
        if (exit_status /= 0) then
            write(output_unit, '(a)') 'No CUDA devices available to FortAI.'
        end if
    end subroutine print_devices

    subroutine print_usage()
        write(output_unit, '(a)') 'fortai-server [options]'
        write(output_unit, '(a)') '  -m, --model PATH             GGUF model'
        write(output_unit, '(a)') '  --host HOST                  bind address (default 127.0.0.1)'
        write(output_unit, '(a)') '  --port PORT                  listen port (default 8080)'
        write(output_unit, '(a)') '  -c, --ctx-size N             context size (0=model default)'
        write(output_unit, '(a)') '  -t, --threads N              CPU threads'
        write(output_unit, '(a)') '  -ngl, --n-gpu-layers N       GPU layers (0=CPU)'
        write(output_unit, '(a)') '  --main-gpu N / -mg           primary GPU'
        write(output_unit, '(a)') '  --alias NAME / -a            served model id (default qwen)'
        write(output_unit, '(a)') '  --parallel N / -np N         resident sequence slots'
        write(output_unit, '(a)') '  -b, --batch-size N           logical prompt batch limit'
        write(output_unit, '(a)') '  -ub, --ubatch-size N         physical prompt microbatch limit'
        write(output_unit, '(a)') '  --flash-attn MODE / -fa      attention mode'
        write(output_unit, '(a)') '  --tensor-split A,B / -ts     two-GPU tensor fractions'
        write(output_unit, '(a)') '  --split-mode MODE / -sm      none, layer, row, or tensor policy'
        write(output_unit, '(a)') '  --model-draft PATH / -md     speculative draft/MTP model'
        write(output_unit, '(a)') '  --spec-type TYPE             speculative mode (draft-mtp)'
        write(output_unit, '(a)') '  --spec-draft-n-max N         maximum speculative draft tokens'
        write(output_unit, '(a)') '  --cache-type-k TYPE          K cache type'
        write(output_unit, '(a)') '  --cache-type-v TYPE          V cache type'
        write(output_unit, '(a)') '  --cache-type-k-draft TYPE    draft K cache type'
        write(output_unit, '(a)') '  --cache-type-v-draft TYPE    draft V cache type'
        write(output_unit, '(a)') '  --cache-ram MB               host KV cache budget'
        write(output_unit, '(a)') '  --fit MODE / -fit MODE       VRAM fit policy'
        write(output_unit, '(a)') '  --cache-reuse N              KV cache reuse threshold'
        write(output_unit, '(a)') '  --n-cpu-moe N                CPU MoE expert count'
        write(output_unit, '(a)') '  --mmproj PATH                multimodal projector'
        write(output_unit, '(a)') '  --mmproj-device DEVICE       multimodal projector device'
        write(output_unit, '(a)') '  --mmproj-offload/--no-mmproj-offload'
        write(output_unit, '(a)') '  --load-mode MODE / -lm       auto, none, mmap, mlock, mmap+mlock, or dio'
        write(output_unit, '(a)') '  --threads-http N             HTTP worker threads'
        write(output_unit, '(a)') '  --api-prefix PREFIX          serve all endpoints below PREFIX'
        write(output_unit, '(a)') '  --path PATH                  serve static files from PATH'
        write(output_unit, '(a)') '  --cors-origins ORIGINS       CORS origin allow-list'
        write(output_unit, '(a)') '  --cors-methods METHODS       CORS method allow-list'
        write(output_unit, '(a)') '  --cors-headers HEADERS       CORS header allow-list'
        write(output_unit, '(a)') '  --api-key KEY                Bearer key (also LLAMA_API_KEY)'
        write(output_unit, '(a)') '  --api-key-file PATH          one Bearer key per line'
        write(output_unit, '(a)') '  --reasoning-budget N         thinking token budget'
        write(output_unit, '(a)') '  --chat-template-kwargs JSON  template defaults (JSON object)'
        write(output_unit, '(a)') '  --temp N                     default sampling temperature'
        write(output_unit, '(a)') '  --top-k N / --top-p N        default sampling cutoffs'
        write(output_unit, '(a)') '  --no-webui                   disable the embedded web UI'
        write(output_unit, '(a)') '  All current llama.cpp server options and LLAMA_ARG_* aliases are accepted.'
    end subroutine print_usage

end program fortai_server
