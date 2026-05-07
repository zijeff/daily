`timescale 1ns/1ns 
module sccomp_tb();
   reg    clk, rstn;
   reg  [4:0] reg_sel;
   wire [31:0] reg_data;
   integer sim_time;
   reg test_done;

   // instantiation of sccomp
   sccomp sccomp(.clk(clk), .rstn(rstn), .reg_sel(reg_sel), .reg_data(reg_data));

   initial begin
     // input instructions for simulation, rv32_sc_sim
      $readmemh("rv32_sc_sim.dat", sccomp.U_imem.RAM, 0, 32);

      clk = 1;
      rstn = 1;
      test_done = 0;
      #10 ;
      rstn = 0;
      reg_sel = 24;
   end

   initial begin
      sim_time = 400;
      if (!$value$plusargs("SIM_TIME=%d", sim_time))
         sim_time = 400;

      $dumpfile("sccomp_tb.vcd");
      $dumpvars(0, sccomp_tb);

      #sim_time;
      if (!test_done) begin
         $display("\nFAIL: simulation timed out at %0d ns before andi/bne sample completed", sim_time);
         $fatal;
      end
   end

   initial begin
      wait (sccomp.PC == 32'h0000_0080);
      #1;

      if (sccomp.U_SCCPU.U_RF.rf[21] !== 32'h12345000) begin
         $display("FAIL: andi result incorrect, x21=%h", sccomp.U_SCCPU.U_RF.rf[21]);
         $fatal;
      end
      if (sccomp.U_SCCPU.U_RF.rf[22] !== 32'h00000001) begin
         $display("FAIL: bne not-taken path incorrect, x22=%h", sccomp.U_SCCPU.U_RF.rf[22]);
         $fatal;
      end
      if (sccomp.U_SCCPU.U_RF.rf[23] !== 32'h00000000) begin
         $display("FAIL: bne taken path did not skip write, x23=%h", sccomp.U_SCCPU.U_RF.rf[23]);
         $fatal;
      end
      if (sccomp.U_SCCPU.U_RF.rf[24] !== 32'h00000002) begin
         $display("FAIL: bne branch target incorrect, x24=%h", sccomp.U_SCCPU.U_RF.rf[24]);
         $fatal;
      end

      test_done = 1;
      $display("\nPASS: default sample verified andi and bne");
      $finish;
   end
   
   always begin
      #(5) clk = ~clk;
   end
   
endmodule
