from emulator.config import Config
from emulator.profiler import Profiler
from emulator.static_profiler import StaticProfiler
from emulator.empty_profiler import EmptyProfiler


class ProfilerFactory:
    @classmethod
    def generate_profiler(cls, config: Config) -> Profiler:
        cl_type = config.cl_type

        if cl_type == "STATIC":
            profiler = StaticProfiler(config=config)
        elif cl_type == "NONE":
            profiler = EmptyProfiler(config=config)
        else:
            raise ValueError(f"invalid cl_type: {cl_type}")

        return profiler