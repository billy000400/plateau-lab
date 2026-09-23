#!/usr/bin/env sh
# Copy the application, run it over SSH, and forward its local web port.
set -eu
usage() {
    echo "Usage: $0 USER@SERVER [auto|cu126|cu128|cu129|rocm6.4|cpu]"
    echo "An SSH-config host alias also works. Default local browser port: 8766."
}
if [ "${1:-}" = "--help" ]; then usage; exit 0; fi
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then usage; exit 2; fi
remote_target=$1
case "$remote_target" in
    ''|-*|*[!a-zA-Z0-9_.@-]*)
        echo "Use USER@SERVER or an SSH-config host alias (put custom ports/keys in SSH config)."
        exit 2 ;;
esac
torch_build=${2:-auto}
case "$torch_build" in
    auto) remote_launch='./start.sh' ;;
    cu126|cu128|cu129|rocm6.4|cpu)
        remote_launch="PLATEAU_TORCH_INDEX_URL=https://download.pytorch.org/whl/$torch_build ./start.sh" ;;
    *) usage; exit 2 ;;
esac
browser_port=${PLATEAU_LOCAL_PORT:-8766}
case "$browser_port" in ''|*[!0-9]*) echo 'PLATEAU_LOCAL_PORT must be a port number.'; exit 2 ;; esac
if [ "$browser_port" -lt 1024 ] || [ "$browser_port" -gt 65535 ]; then
    echo 'PLATEAU_LOCAL_PORT must be between 1024 and 65535.'
    exit 2
fi
for program in ssh rsync; do
    command -v "$program" >/dev/null 2>&1 || { echo "$program is required on this computer."; exit 1; }
done
cd -- "$(dirname -- "$0")"
echo "Copying Plateau Lab code to $remote_target:~/plateau-lab/ …"
# Only application code/docs/assets travel. Remote models, examples and environments
# persist independently. No deletion or process termination is performed remotely.
rsync -az --include='static/***' --include='*.py' --include='*.md' \
    --include='*.sh' --include='*.command' --include='requirements.txt' --exclude='*' \
    ./ "$remote_target:plateau-lab/"
echo "Starting the remote app. First launch may take a few minutes to install packages."
echo "When the server is ready, open http://127.0.0.1:$browser_port/ on this computer."
echo "Keep this Terminal window open; press Ctrl+C to stop this session."
exec ssh -t -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L "127.0.0.1:$browser_port:127.0.0.1:8765" "$remote_target" \
    "cd \"\$HOME/plateau-lab\" && $remote_launch"
