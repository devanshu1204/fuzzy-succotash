from typing import Optional

import httpx


def _inject_user_id(params: Optional[dict], user_id: str) -> dict:
    p = dict(params or {})
    p["user_id"] = user_id
    return p


class AsyncCrudClient(httpx.AsyncClient):
    def __init__(self, base_url: str, user_id: str, **kwargs):
        super().__init__(base_url=base_url, **kwargs)
        self._user_id = user_id

    async def request(self, method, url, *, params=None, **kwargs):
        return await super().request(
            method, url, params=_inject_user_id(params, self._user_id), **kwargs
        )


class CrudClient(httpx.Client):
    def __init__(self, base_url: str, user_id: str, **kwargs):
        super().__init__(base_url=base_url, **kwargs)
        self._user_id = user_id

    def request(self, method, url, *, params=None, **kwargs):
        return super().request(
            method, url, params=_inject_user_id(params, self._user_id), **kwargs
        )
