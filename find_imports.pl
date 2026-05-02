use strict;
use warnings;
use File::Find;

my %modules;

find(
    sub {
        return unless -f $_;                 # 仅处理文件
        return unless $_ =~ /\.py$/i;        # 匹配.py后缀（不区分大小写）

        open my $fh, '<', $_ or do {
            warn "无法打开文件 $_: $!";
            return;
        };

        while (my $line = <$fh>) {
            chomp $line;
            $line =~ s/#.*//;                # 去除注释
            $line =~ s/^\s+|\s+$//g;         # 去除首尾空格

            # 处理import语句（如：import os, sys as system）
            if ($line =~ /^\s*import\s+(.+)/) {
                my @imports = split(/,\s*/, $1);
                foreach my $item (@imports) {
                    $item =~ s/\s+as\s+.*//; # 去除别名
                    my ($module) = split(/\./, $item);
                    next if $module =~ /^\./; # 跳过相对导入
                    $modules{$module} = 1 if $module;
                }
            }
            # 处理from...import语句（如：from django.conf import settings）
            elsif ($line =~ /^\s*from\s+([^\s]+)\s+import/) {
                my $module_path = $1;
                my ($module) = split(/\./, $module_path);
                next if $module =~ /^\./;     # 跳过相对导入
                $modules{$module} = 1 if $module;
            }
        }

        close $fh;
    },
    '.'  # 从当前目录开始遍历
);

# 按字母序输出模块名
foreach my $mod (sort keys %modules) {
    print "$mod\n";
}

