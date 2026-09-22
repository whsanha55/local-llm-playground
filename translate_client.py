"""gemma 번역 서버용 CLI 클라이언트 — 파일 전송 → 진행 폴링 → 결과 저장.

Windows 등 원격 PC에서 사용. 표준라이브러리만 사용(pip 설치 불필요).

사용:
  python translate_client.py --server http://192.168.0.10:8300 자막.srt
  python translate_client.py 자막.srt                      # GEMMA_SERVER 환경변수 사용

동작:
  1) POST /translate 로 파일 업로드(job 등록)
  2) GET /translate/jobs/{id} 를 5초마다 폴링하며 진행 표시
  3) 완료 시 입력 파일 옆에 <이름>.ko<확장자> 로 저장
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

POLL_SECONDS = 5


def post_file(server: str, path: str, model: str = None) -> dict:
    """multipart/form-data 로 업로드 후 {job_id, chunks} 반환."""
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    with open(path, "rb") as f:
        data = f.read()
    parts = []
    if model:
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'.encode()
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    parts.append(data)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        server.rstrip("/") + "/translate", data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.load(res)


def get_job(server: str, job_id: str) -> dict:
    with urllib.request.urlopen(
        f"{server.rstrip('/')}/translate/jobs/{job_id}", timeout=10
    ) as res:
        return json.load(res)


def main():
    ap = argparse.ArgumentParser(description="gemma 번역 서버 클라이언트")
    ap.add_argument("file", help="번역할 .srt/.txt 파일")
    ap.add_argument("--server", default=os.environ.get("GEMMA_SERVER", "http://127.0.0.1:8300"),
                    help="서버 주소 (기본: GEMMA_SERVER env 또는 http://127.0.0.1:8300)")
    ap.add_argument("--model", default=None, help="모델 id (기본: 서버 기본 모델)")
    args = ap.parse_args()

    if not os.path.isfile(args.file):
        sys.exit(f"파일이 없습니다: {args.file}")

    print(f"업로드 중: {args.file} → {args.server}")
    reg = post_file(args.server, args.file, args.model)
    job_id, total = reg["job_id"], reg["chunks"]
    print(f"작업 등록: {job_id} (청크 {total}개) — 취소하려면 서버 UI 또는 "
          f"DELETE {args.server}/translate/jobs/{job_id}")

    t0 = time.time()
    while True:
        time.sleep(POLL_SECONDS)
        try:
            job = get_job(args.server, job_id)
        except urllib.error.URLError as e:
            print(f"폴링 오류({e}) — 재시도"); continue
        state = job["state"]
        if state == "queued":
            print(f"\r대기 중… (앞에 {job['wait_ahead']}개)", end="", flush=True)
        elif state == "running":
            print(f"\r{job['done_chunks']}/{job['chunks']} 청크 · "
                  f"{job['seconds']:.0f}s", end="", flush=True)
        elif state == "done":
            print()
            root, ext = os.path.splitext(args.file)
            out = f"{root}.ko{ext}"
            with open(out, "w", encoding="utf-8") as f:
                f.write(job["translation"])
            skipped = len(job.get("untranslated") or [])
            note = f" (미번역 {skipped}블록)" if skipped else ""
            print(f"완료: {out} · {job['seconds']}s · {job['blocks']}블록{note}")
            return
        else:  # cancelled / error
            print()
            sys.exit(f"작업 {state}: {job.get('error', '')}")


if __name__ == "__main__":
    main()
