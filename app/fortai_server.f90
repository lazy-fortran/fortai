program fortai_server
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char
    use, intrinsic :: iso_fortran_env, only: error_unit, output_unit
    use fortai_native_service, only: fortai_native_service_close, fortai_native_service_init
    use fortai_string, only: string_t
    implicit none

    type :: server_config_t
        type(string_t) :: model
        type(string_t) :: host
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
        integer :: device_length, model_length

        call config%host%set('127.0.0.1')
        call get_environment_variable('FORTAI_SERVER_MODEL', model_env, length=model_length)
        if (model_length > 0 .and. model_length <= len(model_env)) call config%model%set(model_env(:model_length))
        okay = .true.; show_help = .false.; show_version = .false.
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
                i = i + 1; if (i > count .or. .not. parse_integer(argument_at(i), 1, 65535, value)) then; okay = .false.; return; end if
                config%port = value
            case ('-c', '--ctx-size')
                i = i + 1; if (i > count .or. .not. parse_integer(argument_at(i), 128, 2**20, value)) then; okay = .false.; return; end if
                config%context_size = value
            case ('-t', '--threads')
                i = i + 1; if (i > count .or. .not. parse_integer(argument_at(i), 1, 4096, value)) then; okay = .false.; return; end if
                config%threads = value
            case ('-ngl', '--n-gpu-layers', '--gpu-layers')
                i = i + 1; if (i > count .or. .not. parse_integer(argument_at(i), 0, 8192, value)) then; okay = .false.; return; end if
                config%gpu_layers = value
            case ('--main-gpu')
                i = i + 1; if (i > count .or. .not. parse_integer(argument_at(i), 0, 255, value)) then; okay = .false.; return; end if
                config%main_gpu = value
            case ('--parallel', '--tensor-split', '--split-mode', '--model-draft', '--spec-type', '--spec-draft-n-max', &
                    '--flash-attn', '--cache-type-k', '--cache-type-v', '--batch-size', '-b', '--ubatch-size', '-ub')
                i = i + 1; if (i > count) then; okay = .false.; return; end if
                call argument_text_at(i, value_text)
                call set_option_environment(option_text, value_text)
            case ('--jinja', '--no-jinja', '--no-mmap', '--mlock', '--no-kv-offload')
                continue
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
            if (trim(device(:device_length)) == 'cpu' .or. trim(device(:device_length)) == 'host') config%gpu_layers = 0
        end if
        if (config%model%length() == 0) okay = .false.
    end subroutine parse_arguments

    subroutine configure_environment(config)
        type(server_config_t), intent(in) :: config
        call set_environment('FORTAI_LLAMA_FASTPATH', 'native')
        call set_environment('OMP_NUM_THREADS', int_text(config%threads))
        call set_environment('FORTAI_GPU_LAYERS', int_text(config%gpu_layers))
        call set_environment('FORTAI_ENABLE_CUDA_GRAPH', '1')
    end subroutine configure_environment

    subroutine set_option_environment(option, value)
        character(len=*), intent(in) :: option, value
        character(len=32) :: name
        select case (option)
        case ('--parallel'); name = 'FORTAI_PARALLEL'
        case ('--tensor-split'); name = 'FORTAI_TENSOR_SPLIT'
        case ('--split-mode'); name = 'FORTAI_SPLIT_MODE'
        case ('--model-draft'); name = 'FORTAI_DRAFT_MODEL'
        case ('--spec-type'); name = 'FORTAI_SPEC_TYPE'
        case ('--spec-draft-n-max'); name = 'FORTAI_SPEC_DRAFT_N_MAX'
        case ('--flash-attn'); name = 'FORTAI_FLASH_ATTN'
        case ('--cache-type-k'); name = 'FORTAI_CACHE_TYPE_K'
        case ('--cache-type-v'); name = 'FORTAI_CACHE_TYPE_V'
        case ('--ubatch-size', '-ub'); name = 'FORTAI_UBATCH'
        case default; name = 'FORTAI_BATCH'
        end select
        call set_environment(trim(name), value)
    end subroutine set_option_environment

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
        write(output_unit, '(a)') '  --main-gpu N                 primary GPU'
        write(output_unit, '(a)') '  --parallel N                 resident sequence slots'
        write(output_unit, '(a)') '  --flash-attn on|off|auto     attention mode'
    end subroutine print_usage

end program fortai_server
