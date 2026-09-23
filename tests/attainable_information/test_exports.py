import attainable_information as ai
from attainable_information import bounds, recoverability


def test_module_all_lists_are_exported_at_top_level():
    for mod in (recoverability, bounds):
        for name in mod.__all__:
            assert name in ai.__all__, name
            assert getattr(ai, name) is getattr(mod, name)


def test_public_functions_are_in_all():
    for mod in (recoverability, bounds):
        public = {k for k, v in vars(mod).items() if callable(v) and not k.startswith("_") and v.__module__ == mod.__name__}
        assert public <= set(mod.__all__), public - set(mod.__all__)
