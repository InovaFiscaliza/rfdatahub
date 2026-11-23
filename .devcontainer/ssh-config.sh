#!/bin/sh
set -eu

install -d -m 755 /etc/ssh/sshd_config.d
cat > /etc/ssh/sshd_config.d/00-devcontainer.conf <<'EOF'
Port 2222
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
AllowUsers dev
EOF
chmod 644 /etc/ssh/sshd_config.d/00-devcontainer.conf

ssh-keygen -A
/usr/sbin/sshd -t
exec /usr/sbin/sshd

touch ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys