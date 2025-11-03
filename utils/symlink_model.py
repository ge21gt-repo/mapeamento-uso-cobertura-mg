import os
import shutil

def create_symlink(input, output):
    src = os.path.abspath(input)
    dst = os.path.abspath(output)

    if not os.path.exists(src):
        raise FileNotFoundError(f"Origem não encontrada: {src}")

    # Remove link ou diretório existente
    if os.path.lexists(dst):
        if os.path.isdir(dst) and not os.path.islink(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)

    try:
        os.symlink(src, dst, target_is_directory=True)
        print("✅ Symlink criado com sucesso!")
    except OSError as e:
        print("Erro ao criar symlink:", e)
