python  := 'uv run python3'
file_nc := '../data/D20230803-T230004.nc'

_all:
    @echo "Try justjus --list?"

test:
    time {{python}} src/main.py {{file_nc}}

profile:
    time {{python}} -m cProfile src/main.py {{file_nc}} > OUT.prof

test-detect:
    time {{python}} src/netcdf2dets.py {{file_nc}}

test-track:
    time {{python}} src/track.py {{file_nc}}

profile-detect:
    {{python}} -m cProfile src/netcdf2dets.py {{file_nc}} > OUT_d.prof

profile-track:
    time {{python}} -m cProfile src/track.py {{file_nc}} > OUT_t.profile
