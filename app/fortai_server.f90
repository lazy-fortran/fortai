program fortai_server
    use, intrinsic :: iso_c_binding, only: c_char, c_int
    use, intrinsic :: iso_fortran_env, only: error_unit, output_unit
    use fortai_native_service, only: fortai_native_service_close, fortai_native_service_init
    use fortai_string, only: string_t
    implicit none

    type :: server_config_t
        type(string_t) :: model
        type(string_t) :: host
        type(string_t) :: alias
        integer :: port = 8080
        integer :: context_size = 4096
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
    logical :: okay, show_help, show_version
    integer(c_int) :: status, vocab, layers
    integer :: require_cuda
    character(kind=c_char), allocatable :: cmodel(:), chost(:)
    character(len=32) :: number

    call parse_arguments(config, okay, show_help, show_version)
    if (show_help) then
        call print_usage()
        stop
    end if
    if (show_version) then
        write(output_unit, '(a)') 'fortai-server 0.1 (FortAI-owned native Fortran runtime)'
        stop
    end if
    if (.not. okay) then
        call print_usage()
        error stop 2
    end if
    if (config%threads <= 0) config%threads = int(fortai_server_online_cpus())
    if (config%threads <= 0) config%threads = 1
    call configure_environment(config)

    allocate(cmodel(config%model%length() + 1), chost(config%host%length() + 1))
    call config%model%to_c(cmodel, size(cmodel))
    call config%host%to_c(chost, size(chost))
    require_cuda = merge(1, 0, config%gpu_layers > 0)
    write(number, '(i0)') config%gpu_layers
    write(error_unit, '(a)') 'FORTAI_SERVER_BACKEND=fortai model=' // config%model%as_character() // &
        ' alias=' // config%alias%as_character() // &
        ' host=' // config%host%as_character() // ' port=' // trim(int_text(config%port)) // &
        ' threads=' // trim(int_text(config%threads)) // ' gpu_layers=' // trim(number)
    status = fortai_native_service_init(cmodel, int(config%context_size, c_int), int(config%threads, c_int), &
        int(config%gpu_layers, c_int), int(config%main_gpu, c_int), int(require_cuda, c_int), vocab, layers)
    if (status /= 0_c_int) then
        write(error_unit, '(a)') 'fortai-server: FortAI model context creation failed'
        error stop 1
    end if
    write(error_unit, '(a)') 'FORTAI_SERVER_READY=1 vocab=' // trim(int_text(int(vocab))) // &
        ' layers=' // trim(int_text(int(layers)))
    status = fortai_http_transport_run(chost, int(config%port, c_int), cmodel, &
        int(merge(1, 0, config%gpu_layers > 0), c_int))
    call fortai_native_service_close()
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

    subroutine parse_arguments(config, okay, show_help, show_version)
        type(server_config_t), intent(inout) :: config
        logical, intent(out) :: okay, show_help, show_version
        integer :: i, count, value
        type(string_t) :: argument, option
        character(len=:), allocatable :: option_text, value_text
        character(len=16) :: device
        character(len=4096) :: model_env
        character(len=256) :: alias_env
        integer :: device_length, model_length, alias_length

        okay = .true.; show_help = .false.; show_version = .false.
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
            case ('--help', '-h')
                show_help = .true.; return
            case ('--version')
                show_version = .true.; return
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
                if (.not. parse_integer(argument_at(i), 128, 2**20, value)) then; okay = .false.; return; end if
                config%context_size = value
            case ('-t', '--threads')
                i = i + 1
                if (i > count) then; okay = .false.; return; end if
                if (.not. parse_integer(argument_at(i), 1, 4096, value)) then; okay = .false.; return; end if
                config%threads = value
            case ('-ngl', '--n-gpu-layers', '--gpu-layers')
                i = i + 1
                if (i > count) then; okay = .false.; return; end if
                if (.not. parse_integer(argument_at(i), 0, 8192, value)) then; okay = .false.; return; end if
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
                    '--n-cpu-moe', '--ncmoe', '--reasoning-budget', '--mmproj', '-mm', '--rpc', '--threads-http', &
                    '--temp', '--temperature', '--top-k', '--top-p', '--min-p', '--repeat-penalty', &
                    '--presence-penalty', '--frequency-penalty', '--repeat-last-n', '--seed', '--reasoning-format', &
                    '--reasoning', '--reasoning-effort', '--threads-batch', '-tb', '--n-predict', '--predict', '-n', &
                    '--max-tokens', '--device')
                i = i + 1; if (i > count) then; okay = .false.; return; end if
                call argument_text_at(i, value_text)
                call set_option_environment(option_text, value_text)
            case ('--alias', '-a', '--served-model-name')
                i = i + 1; if (i > count) then; okay = .false.; return; end if
                call argument_text_at(i, value_text)
                call config%alias%set(value_text)
            case ('--jinja', '--no-jinja', '--mmap', '--no-mmap', '--mlock', '--kv-offload', '--no-kv-offload', &
                    '--context-shift', '--no-context-shift', '--metrics', '--log-timestamps', '--mmproj-offload', &
                    '--no-mmproj-offload', '--webui', '--no-ui', '--no-webui', '--op-offload', '--no-op-offload', &
                    '--reasoning-preserve', '--no-reasoning-preserve')
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
        if (device_length > 0) then
            select case (trim(device(:device_length)))
            case ('cpu', 'host', 'none')
                config%gpu_layers = 0
            end select
        end if
        if (config%model%length() == 0) okay = .false.
    end subroutine parse_arguments

    subroutine configure_environment(config)
        type(server_config_t), intent(in) :: config
        call set_environment('FORTAI_LLAMA_FASTPATH', 'native')
        call set_environment('OMP_NUM_THREADS', int_text(config%threads))
        call set_environment('FORTAI_GPU_LAYERS', int_text(config%gpu_layers))
        call set_environment('FORTAI_ENABLE_CUDA_GRAPH', '1')
        call set_environment('FORTAI_SERVER_ALIAS', config%alias%as_character())
    end subroutine configure_environment

    subroutine set_option_environment(option, value)
        character(len=*), intent(in) :: option, value
        character(len=32) :: name
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
        case ('--n-cpu-moe', '--ncmoe'); name = 'FORTAI_N_CPU_MOE'
        case ('--reasoning-budget'); name = 'FORTAI_REASONING_BUDGET'
        case ('--reasoning-effort'); name = 'FORTAI_REASONING_EFFORT'
        case ('--reasoning'); name = 'FORTAI_REASONING'
        case ('--mmproj', '-mm'); name = 'FORTAI_MMPROJ'
        case ('--rpc'); name = 'FORTAI_RPC'
        case ('--threads-http'); name = 'FORTAI_THREADS_HTTP'
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
        case ('--seed'); name = 'FORTAI_SEED'
        case ('--reasoning-format'); name = 'FORTAI_REASONING_FORMAT'
        case ('--threads-batch', '-tb'); name = 'FORTAI_THREADS_BATCH'
        case ('--n-predict', '--predict', '-n', '--max-tokens'); name = 'FORTAI_MAX_TOKENS'
        case ('--device'); name = 'FORTAI_SERVER_DEVICE'
        case default; name = 'FORTAI_OPTION'
        end select
        call set_environment(trim(name), value)
    end subroutine set_option_environment

    subroutine set_flag_environment(option)
        character(len=*), intent(in) :: option
        character(len=32) :: name, value
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
        case ('--webui'); name = 'FORTAI_NO_WEBUI'; value = '0'
        case ('--no-ui', '--no-webui'); name = 'FORTAI_NO_WEBUI'; value = '1'
        case ('--op-offload'); name = 'FORTAI_NO_OP_OFFLOAD'; value = '0'
        case ('--no-op-offload'); name = 'FORTAI_NO_OP_OFFLOAD'; value = '1'
        case ('--reasoning-preserve'); name = 'FORTAI_REASONING_PRESERVE'; value = '1'
        case ('--no-reasoning-preserve'); name = 'FORTAI_REASONING_PRESERVE'; value = '0'
        case default; return
        end select
        call set_environment(trim(name), trim(value))
    end subroutine set_flag_environment

    subroutine load_environment_defaults(config, okay)
        type(server_config_t), intent(inout) :: config
        logical, intent(out) :: okay
        character(len=4096) :: value
        integer :: length

        okay = .true.
        call load_text_environment('FORTAI_SERVER_HOST', 'LLAMACPP_HOST', config%host)
        call load_integer_environment('FORTAI_SERVER_PORT', 'LLAMACPP_PORT', 1, 65535, config%port, okay)
        call load_integer_environment('FORTAI_CONTEXT', 'LLAMACPP_CONTEXT', 128, 2**20, config%context_size, okay)
        call load_integer_environment('FORTAI_THREADS', 'LLAMACPP_THREADS', 0, 4096, config%threads, okay)
        call load_integer_environment('FORTAI_GPU_LAYERS', 'LLAMACPP_GPU_LAYERS', 0, 8192, config%gpu_layers, okay)
        call load_integer_environment('FORTAI_MAIN_GPU', 'LLAMACPP_MAIN_GPU', 0, 255, config%main_gpu, okay)
        if (.not. okay) return
        value = ''; call get_environment_variable('FORTAI_SERVER_ALIAS', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_SERVED_ALIAS', value, length=length)
        if (length <= 0) call get_environment_variable('LLAMACPP_MODEL_ALIAS', value, length=length)
        if (length > 0 .and. length <= len(value)) call config%alias%set(value(:length))
        call load_text_environment('FORTAI_ALIAS', 'LLAMACPP_SERVED_ALIAS', config%alias)
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
    end subroutine load_environment_defaults

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
        call import_ui_alias()
    end subroutine import_llama_environment

    subroutine import_environment_alias(target_name, source_name)
        character(len=*), intent(in) :: target_name, source_name
        character(len=4096) :: value
        integer :: length

        value = ''
        call get_environment_variable(target_name, value, length=length)
        if (length > 0) return
        value = ''
        call get_environment_variable(source_name, value, length=length)
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

    subroutine argument_text_at(index, text)
        integer, intent(in) :: index
        character(len=:), allocatable, intent(out) :: text
        type(string_t) :: argument
        argument = argument_at(index)
        text = argument%as_character()
    end subroutine argument_text_at

    subroutine print_usage()
        write(output_unit, '(a)') 'fortai-server [options]'
        write(output_unit, '(a)') '  -m, --model PATH             GGUF model'
        write(output_unit, '(a)') '  --host HOST                  bind address (default 127.0.0.1)'
        write(output_unit, '(a)') '  --port PORT                  listen port (default 8080)'
        write(output_unit, '(a)') '  -c, --ctx-size N             context size'
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
        write(output_unit, '(a)') '  --mmproj-offload/--no-mmproj-offload'
        write(output_unit, '(a)') '  --threads-http N             HTTP worker threads'
        write(output_unit, '(a)') '  --reasoning-budget N         thinking token budget'
        write(output_unit, '(a)') '  --temp N                     default sampling temperature'
        write(output_unit, '(a)') '  --top-k N / --top-p N        default sampling cutoffs'
        write(output_unit, '(a)') '  --no-webui                   disable the embedded web UI'
    end subroutine print_usage

end program fortai_server
