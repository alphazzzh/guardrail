"""HTTP 客户端池管理"""
import atexit
import logging
import threading
from typing import Dict, Optional, Tuple

import httpx

logger = logging.getLogger("safeguard_system")

_HTTPX_CLIENTS: Dict[Tuple[float, float, int, int, float, float], httpx.Client] = {}
_ASYNC_HTTPX_CLIENTS: Dict[Tuple[float, float, int, int, float, float], httpx.AsyncClient] = {}
_HTTPX_CLIENTS_LOCK = threading.Lock()


def get_shared_http_client(
    connect_timeout: float,
    read_timeout: float,
    max_connections: int = 100,
    max_keepalive: int = 20,
    write_timeout: float = 5.0,
    pool_timeout: Optional[float] = None,
) -> httpx.Client:
    """获取共享的同步 HTTP 客户端"""
    pool_timeout = read_timeout if pool_timeout is None else pool_timeout
    key = (
        float(connect_timeout),
        float(read_timeout),
        int(max_connections),
        int(max_keepalive),
        float(write_timeout),
        float(pool_timeout),
    )
    with _HTTPX_CLIENTS_LOCK:
        cli = _HTTPX_CLIENTS.get(key)
        if cli is not None:
            return cli

        timeout = httpx.Timeout(
            connect=connect_timeout, read=read_timeout, write=write_timeout, pool=pool_timeout
        )
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_keepalive)
        cli = httpx.Client(timeout=timeout, limits=limits)
        _HTTPX_CLIENTS[key] = cli
        return cli


def get_shared_async_http_client(
    connect_timeout: float,
    read_timeout: float,
    max_connections: int = 100,
    max_keepalive: int = 20,
    write_timeout: float = 5.0,
    pool_timeout: Optional[float] = None,
) -> httpx.AsyncClient:
    """获取共享的异步 HTTP 客户端"""
    pool_timeout = read_timeout if pool_timeout is None else pool_timeout
    key = (
        float(connect_timeout),
        float(read_timeout),
        int(max_connections),
        int(max_keepalive),
        float(write_timeout),
        float(pool_timeout),
    )
    with _HTTPX_CLIENTS_LOCK:
        cli = _ASYNC_HTTPX_CLIENTS.get(key)
        if cli is not None:
            return cli

        timeout = httpx.Timeout(
            connect=connect_timeout, read=read_timeout, write=write_timeout, pool=pool_timeout
        )
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_keepalive)
        cli = httpx.AsyncClient(timeout=timeout, limits=limits)
        _ASYNC_HTTPX_CLIENTS[key] = cli
        return cli


def close_all_http_clients() -> None:
    """关闭所有共享的同步 HTTP 客户端（atexit 安全，仅处理同步客户端）"""
    with _HTTPX_CLIENTS_LOCK:
        sync_clients = list(_HTTPX_CLIENTS.values())
        _HTTPX_CLIENTS.clear()

    for client in sync_clients:
        try:
            client.close()
        except Exception:
            logger.debug("close shared sync httpx client failed", exc_info=True)


async def aclose_all_http_clients() -> None:
    """关闭所有共享的异步 HTTP 客户端（需在事件循环中调用，在 FastAPI lifespan 结束时使用）"""
    with _HTTPX_CLIENTS_LOCK:
        async_clients = list(_ASYNC_HTTPX_CLIENTS.values())
        _ASYNC_HTTPX_CLIENTS.clear()

    for client in async_clients:
        try:
            await client.aclose()
        except Exception:
            logger.debug("close shared async httpx client failed", exc_info=True)


@atexit.register
def _cleanup_http_clients() -> None:
    """进程退出时清理 HTTP 客户端"""
    close_all_http_clients()


def get_async_httpx_client(
    connect_timeout: float = 3.0,
    read_timeout: float = 30.0,
    max_connections: int = 50,
    max_keepalive: int = 20,
) -> httpx.AsyncClient:
    """get_shared_async_http_client 的简化别名"""
    return get_shared_async_http_client(
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_connections=max_connections,
        max_keepalive=max_keepalive,
    )
