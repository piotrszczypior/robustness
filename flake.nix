{
    description = "Robusstness Flake";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    };


    outputs = { self, nixpkgs }:
        let
            system = "x86_64-linux";
            pkgs = import nixpkgs {
                inherit system;
                config.allowUnfree = true;
            };
        in {
            devShells.${system}.default = pkgs.mkShell {
                packages = with pkgs; [
                    python312
                    rclone

                    (pkgs.python312.withPackages (ps: with ps; [
                      gdown
                      ruff
                      tkinter
                      matplotlib
                    ]))
                ];

                shellHook = ''
                    source .venv/bin/activate
                    pip install -r requirements.txt
                '';
            };
        };
}