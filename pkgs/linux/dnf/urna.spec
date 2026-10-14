# the release's static musl binary, repackaged: the same bytes every other
# channel ships, so nothing is compiled, stripped or split here.
%global debug_package %{nil}
%global __strip /bin/true

Name:           urna
Version:        ${version}
Release:        1%{?dist}
Summary:        ${summary}
License:        ${license}
URL:            ${homepage}
Source0:        ${url_x86_64_linux}
Source1:        ${url_aarch64_linux}
ExclusiveArch:  x86_64 aarch64

%description
${description}

%prep
%setup -q -c -T
%ifarch x86_64
tar -xJf %{SOURCE0} --strip-components=1
%endif
%ifarch aarch64
tar -xJf %{SOURCE1} --strip-components=1
%endif

%install
install -Dm0755 urna %{buildroot}%{_bindir}/urna

%files
%license LICENSE
%doc README.md
%{_bindir}/urna
