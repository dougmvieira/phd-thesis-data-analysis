from subprocess import run


def compile_paper(folder, name):
    run(['latexmk', '-f', '-silent', '-pdf', name], cwd=folder)
