python  := 'uv run python3'
file_nc := '../data/D20230803-T230004.nc'

test:
    {{python}} src/netcdf2dets.py {{file_nc}}

profile:
    {{python}} -m cProfile src/netcdf2dets.py {{file_nc}} > OUT.prof

