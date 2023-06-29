function driver_injections_in_bsd_data_gw_transient(goutL)
tcoe=(t0_O3+60); % coalescing time [days]
tcoes=tcoe*86400; % coalescing time [s]
days = 1
t=tcoes:tcoes+days*86400; % time vector [s]
% signal parameters:
tau=1500000;
nbreak=5; % breaking index
fgw0=108.01 % gw frequency
fgw=fgw0*(1 +(t-tcoes)/tau).^(1/(1-nbreak));

H0=(fgw0^2)*1.0e-25; % max signal amplitude
K=1e-5
H=H0*(1+(t-tcoes)/tau).^(2/(1-nbreak));
if iplot==1
    figure,plot((t-min(t))/86400,fgw)
    hold on
    grid on
    xlabel('time [days]')
    ylabel('frequency [Hz]')
    plot((tcoes-min(t))/86400,max(fgw),'*')
    figure,plot(t/86400,H);
    hold on
    grid on
    xlabel('time [days]')
    ylabel('H')
    plot(tcoe,max(H),'*')
    figure,plot((t-min(t))/86400,H.*sin(2*pi*fgw))
end

% sour structure:
sour.type='transient'
sour.a=94.9690
sour.d=-17.3432
sour.v_a=0
sour.v_d=0
sour.pepoch=tcoe;
sour.tcoe=tcoe;%
sour.f0=fgw0
sour.df0=0
sour.ddf0=0
sour.fepoch=tcoe;
sour.ecl=[96.2585 -40.6746]
sour.t00=tcoe
sour.eps=1
sour.eta=-0.8530
sour.psi= 8.9378
sour.h=1.0000e-20 % not used
ssour.nr=1
sour.coor=0
sour.chphase= 0
sour.dfrsim= 0
sour.PH0=0
sour.k=K;
sour.n=nbreak
sour.tau=tau
sour.days=days
% 

tini=tcoe-1; % starting time of BSD selected data
tfin=tini+2;   % stop time of BSD selected data 
band=[107 108] % freq. band to be extracted
RUN='C01_GATED_SUB60HZ_O3' % O3 RUN name
goutLC=cut_bsd(goutL,[tini tfin]) % to select the rigth time  interval

fileSinjL=strcat('bsd_softInj_LL_longtransient_powlaw',RUN,'_',num2str(band(1)),'_',num2str(band(2)),'_','.mat'); % name of the bsd with injected signal

A=1; % to be changed befor ato dd the BSD signal to the BSD data  
gSinjL=0*goutL;
[gsinjL]=bsd_softinj_re_mod_gen(goutLC,sour,A);
gsinjL=bsd_zeroholes(gsinjL); % zeroes in the bsd Sinj file 
save(fileSinjL,'gsinjL','sour'); % save the BSD with the injected signal

% cont=cont_gd(bsd_data_plus_signal) % info on the bsd file ex:
% cont gives:
%           t0: 5.8634e+04
%           inifr: 295
%           bandw: 5
%            v_eq: [337?4 double]
%            p_eq: [337?4 double]
%            Tfft: 1024
%             ant: 'ligol'
%             run: 'O3'
%             cal: 'C01'
%           notes: 'GATED_SUB60HZ'
%       tcreation: '19-Oct-2020 19:32:38'
%     durcreation: 37.5130
%      mi_abs_lev: 6.9382e-04
%           tfstr: [1?1 struct]
%         sb_pars: [1?1 struct]
%            oper: [1?1 struct]
end
