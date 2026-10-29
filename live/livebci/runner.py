"""Launch your pipeline's scripts (record, train, live decode) and stream their output lines to the browser."""
import asyncio
import os


class Runner:
    def __init__(self, scripts, on_line, on_exit, base_dir="."):
        self.scripts, self.on_line, self.on_exit, self.base = scripts, on_line, on_exit, base_dir
        self.procs = {}

    def running(self):
        return sorted(n for n, p in self.procs.items() if p.returncode is None)

    async def start(self, name):
        if name not in self.scripts:
            raise KeyError("no script called %r in the config" % name)
        if name in self.running():
            return
        spec = self.scripts[name]
        cwd = os.path.abspath(os.path.join(self.base, spec.get("cwd", ".")))
        proc = await asyncio.create_subprocess_exec(*spec["cmd"], cwd=cwd, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT)
        self.procs[name] = proc
        asyncio.get_running_loop().create_task(self._pump(name, proc))

    async def _pump(self, name, proc):
        async for raw in proc.stdout:
            self.on_line(name, raw.decode(errors="replace").rstrip())
        code = await proc.wait()
        self.on_exit(name, code)

    async def stop(self, name):
        p = self.procs.get(name)
        if p and p.returncode is None:
            p.terminate()
            try:
                await asyncio.wait_for(p.wait(), 5)
            except asyncio.TimeoutError:
                p.kill()

    async def stop_all(self):
        for n in list(self.procs):
            await self.stop(n)
