__version__ = "0.2.0"
is_simple_core = False

if is_simple_core:

    from dezero.core_simple import (
        Function,
        Variable,
        as_array,
        as_variable,
        no_grad,
        using_config,
    )

else:
    from dezero.core import (
        Function,
        Variable,
        as_array,
        as_variable,
        no_grad,
        using_config,
    )
