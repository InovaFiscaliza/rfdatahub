sudo -n pacman -S --needed --noconfirm base-devel git unixodbc
workdir="$(mktemp -d)"
git clone https://aur.archlinux.org/msodbcsql17.git "$workdir/msodbcsql17"
cd "$workdir/msodbcsql17"
makepkg -si --noconfirm
odbcinst -q -d