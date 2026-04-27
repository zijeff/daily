TOP      := sccomp_tb
SIMV     := $(TOP).out
WAVE     := $(TOP).vcd
SIM_TIME ?= 200

SRCS := \
	$(TOP).v \
	sccomp.v \
	SCCPU.v \
	ctrl.v \
	PC.v \
	NPC.v \
	EXT.v \
	RF.v \
	alu.v \
	dm.v \
	im.v

.PHONY: all compile run wave clean help

all: compile

compile: $(SIMV)

$(SIMV): $(SRCS) ctrl_encode_def.v rv32_sc_sim.dat
	iverilog -g2012 -o $@ $(SRCS)

run: $(SIMV)
	vvp $(SIMV) +SIM_TIME=$(SIM_TIME)

wave: $(SIMV)
	vvp $(SIMV) +SIM_TIME=$(SIM_TIME)
	gtkwave $(WAVE)

clean:
	rm -f $(SIMV) $(WAVE)

help:
	@printf '%s\n' \
	'make compile          Compile the design with iverilog' \
	'make run              Run simulation with vvp' \
	'make wave             Run simulation and open gtkwave' \
	'make clean            Remove generated files' \
	'make run SIM_TIME=300 Override simulation stop time in ns'
