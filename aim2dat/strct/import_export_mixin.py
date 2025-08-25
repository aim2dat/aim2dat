"""Import/export mixing class and decorators."""


def import_method(func):
    """Mark function as import function."""
    func._is_import_method = True
    return func


def export_method(func):
    """Mark function as export function."""
    func._is_export_method = True
    return func


class classproperty:
    """Custom, temporary decorator to depreciate class properties."""

    def __init__(self, func):
        """Initiate class."""
        self.fget = func

    def __get__(self, instance, owner):
        """Get method."""
        from warnings import warn

        warn(
            "This function will be removed soon, please use the `list_*_methods` instead.",
            DeprecationWarning,
            2,
        )
        return self.fget(owner)


class ImportExportMixin:
    """Mixin class for classes with import/export functions."""

    @classmethod
    def list_import_methods(cls) -> list:
        """
        Get a list with the function names of all available import methods.

        Returns
        -------
        list:
            Return a list of all available import methods.
        """
        import_methods = []
        for name, method in cls.__dict__.items():
            if getattr(method, "_is_import_method", False):
                import_methods.append(name)
        return import_methods

    @classproperty
    def import_methods(cls) -> list:
        """list: Return import methods. This property is depreciated and will be removed soon."""
        return cls.list_import_methods()

    @classmethod
    def list_export_methods(cls) -> list:
        """
        Get a list with the function names of all available export methods.

        Returns
        -------
        list:
            Return a list of all available export methods.
        """
        export_methods = []
        for name, method in cls.__dict__.items():
            if getattr(method, "_is_export_method", False):
                export_methods.append(name)
        return export_methods

    @classproperty
    def export_methods(cls) -> list:
        """list: Return export methods. This property is depreciated and will be removed soon."""
        return cls.list_export_methods()
