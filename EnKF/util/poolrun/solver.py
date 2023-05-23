                # calculate the model
                ux,uy,uz = yang_disp(enkf_x, enkf_y, 0,
                                     xs=RP[0+npar*cham,n]*1e3, ys=RP[1+npar*cham,n]*1e3, 
                                     zs=-RP[2+npar*cham,n]*1e3,
                                     a=RP[3+npar*cham,n]*1e3, b=RP[4+npar*cham,n]*1e3, 
                                     theta=RP[5+npar*cham,n], phi=RP[6+npar*cham,n], 
                                     dP=RP[7+npar*cham,n]*sdp*1e6,E=30e9, nu=0.25)