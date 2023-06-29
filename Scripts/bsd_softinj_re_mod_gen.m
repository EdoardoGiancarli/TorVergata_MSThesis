function [out Hp Hc]=bsd_softinj_re_mod_gen(in,sour,A)
% Software injection by reverse of bsd_dopp_sd (to be preferred to bsd_softinj)
%
%    out=bsd_softinj_re(in,sour,A)
% 
%   in     input bsd
%   sour   source structure
%   A      amplitude


N=n_gd(in);
dt=dx_gd(in);
cont=cont_gd(in);
y=y_gd(in);
t0=cont.t0;
inifr=cont.inifr;
ant1=cont.ant;
eval(['ant=' ant1 ';'])

sour=new_posfr(sour,t0);
fr=[sour.f0,sour.df0,sour.ddf0];
f0=sour.f0;

obs=check_nonzero(in,1);
if isfield(sour,'type')
    switch sour.type
        case 'transient'
            f0gw=sour.f0;
            nbreak=sour.n;
            K=sour.k;
            tcoe=sour.tcoe; % sour.fepoch=coalescing time
            Ncoa=round((tcoe-t0)*86400/dt)+1;% N of sample for the coales.
            fgw0=sour.f0; % freq. sour at t0;
            tau=-((fgw0)/2)^(1-nbreak)/(K*(1-nbreak));
            if isfield(sour,'tau')
                tau=sour.tau
            end
            % avoid alias.
            fgw(1:N)=0;
            fgw(Ncoa:N)=(f0gw)*(1 +(0:N-Ncoa)*dt/tau).^(1/(1-nbreak))-inifr;
            jck=find((fgw)>=cont.bandw |fgw <0 );
            fgw(jck)=0;
            %    
            ph0=cumsum(fgw)*2*pi*dt;
            in=edit_gd(in,'y',exp(1j*ph0).*obs');
            
            % effetto einstein
            VPstr=extr_velpos_gd(in);
            [pO v0]=interp_VP(VPstr,sour);
            fr=diff(ph0)/(2*pi);
            fr(length(ph0))=fr(length(ph0)-1); %% CHECK
            
            out=vfs_subhet(in,-fr.*v0') % fr is a vector with the vring frequency
            
            einst=einst_effect(t0:dt/86400:t0+(N-0.5)*dt/86400);
            out=vfs_subhet(out,(einst-1)*(fgw0)); %%% TO CHECK if it's ok Ed
            %fprintf('No Einstein delay!\n');
            
            % signal amplitude:
            sig0=y_gd(out);
            
            sig0(Ncoa:N)=sig0(Ncoa:N).*((1+(0:N-Ncoa)*dt/tau).^(2/(1-nbreak)))'; 
            sig0(1:Ncoa)=0;
            sig0(jck)=0*sig0(jck);
            sig=sig0*0;
        case 'other'
    end

% 5-vect

nsid=10000;
stsub=gmst(t0)+dt*(0:N-1)*(86400/Tsid)/3600; % running Greenwich mean sidereal time 
isub=mod(round(stsub*(nsid-1)/24),nsid-1)+1; % time indexes at which the sidereal response is computed
[~, ~ , ~, ~, sid1 sid2]=check_ps_lf(sour,ant,nsid); % computation of the sidereal response (l0 and l45 only) (other two comm plot in check_ps_lf)
gl0=sid1(isub).*sig0.';
gl45=sid2(isub).*sig0.';
Hp=sqrt(1/(1+sour.eta^2))*(cos(2*sour.psi/180*pi)-1j*sour.eta*sin(2*sour.psi/180*pi)); % estimated + complex amplitude (Eq.2 of ref.1)
Hc=sqrt(1/(1+sour.eta^2))*(sin(2*sour.psi/180*pi)+1j*sour.eta*cos(2*sour.psi/180*pi));
sig=Hp*gl0+Hc*gl45;
sig=sig.';

out=edit_gd(out,'y',A*sig);
out=bsd_zeroholes(out);