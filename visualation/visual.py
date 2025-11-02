def string_report(s):
    print(f"🔹 string input : '{s}'\n")
    methods = [
        ("upper()", s.upper()),
        ("lower()", s.lower()),
        ("capitalize()", s.capitalize()),
        ("title()", s.title()),
        ("swapcase()", s.swapcase()),
        ("strip()", s.strip()),
        ("lstrip()", s.lstrip()),
        ("rstrip()", s.rstrip()),
        ("replace('a','@')", s.replace("a", "@")),
        ("count('a')", s.count('a')),
        ("startswith('a')", s.startswith('a')),
        ("endswith('a')", s.endswith('a')),
        ("find('a')", s.find('a')),
        ("rfind('a')", s.rfind('a')),
        ("index('a')", s.index('a') if 'a' in s else "nothing"),
        ("rindex('a')", s.rindex('a') if 'a' in s else "nothing"),
        ("split()", s.split()),
        ("split('a')", s.split('a')),
        ("rsplit('a')", s.rsplit('a')),
        ("partition('a')", s.partition('a')),
        ("rpartition('a')", s.rpartition('a')),
        ("center(20,'*')", s.center(20, '*')),
        ("ljust(20,'-')", s.ljust(20, '-')),
        ("rjust(20,'-')", s.rjust(20, '-')),
        ("zfill(20)", s.zfill(20)),
        ("isalnum()", s.isalnum()),
        ("isalpha()", s.isalpha()),
        ("isdigit()", s.isdigit()),
        ("islower()", s.islower()),
        ("isupper()", s.isupper()),
        ("istitle()", s.istitle()),
        ("isspace()", s.isspace()),
        ("'!'.join(s)", '!'.join(s)),
        ("encode()", s.encode()),
        ("expandtabs(4)", s.expandtabs(4)),
        ("casefold()", s.casefold())
        ]
    for name, result in methods:
        print(f"{name:<2} = {result}")
string_report("  nazanin  ")
