import sys
import qrcode

print(sys.version)

qrcode.make("some really longer test  de longer passw text 2 "
            "bumusafdsa jl sdff").save("/home/spometun/myqr.png")
