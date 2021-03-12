within ;
model PV
    // Weather data
  parameter String weather_file = "" "Path to weather file";
  // PV generation
  parameter Real n(min=0, unit="1") = 26 "Number of PV modules";
  parameter Real A(min=0, unit="m2") = 1.65 "Net surface area per module";
  parameter Real eta(min=0, max=1, unit="1") = 0.158
    "Module conversion efficiency";
  parameter Real lat(unit="deg") = 37.9 "Latitude";
  parameter Real til(unit="deg") = 10 "Surface tilt";
  parameter Real azi(unit="deg") = 0 "Surface azimuth 0-S, 90-W, 180-N, 270-E ";
  SCooDER.Components.Photovoltaics.Model.PVModule_simple
                  pVModule_simple(
    n=n,
    A=A,
    eta=eta,
    lat=lat,
    til=til,
    azi=azi) annotation (Placement(transformation(extent={{-10,40},{10,60}})));
  Buildings.BoundaryConditions.WeatherData.ReaderTMY3 weaDatInpCon(
      computeWetBulbTemperature=false, filNam=
        "/mnt/shared/utilities/ModelicaModel/weather/flexlab_2020_Jun2Aug.mos")
    "Weather data reader with radiation data obtained from the inputs' connectors"
    annotation (Placement(transformation(extent={{-80,60},{-60,80}})));
  Modelica.Blocks.Interfaces.RealInput scale(start=1, unit="1")
    annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
  Modelica.Blocks.Math.Gain WtokW(k=1/1e3)
    annotation (Placement(transformation(extent={{20,0},{40,20}})));
  Modelica.Blocks.Interfaces.RealOutput P_kw(unit="kW", start=0)
    "Active power"
    annotation (Placement(transformation(extent={{100,0},{120,20}})));
  Modelica.Blocks.Interfaces.RealOutput P(start=0, unit="W")
    "Active power"
    annotation (Placement(transformation(extent={{100,40},{120,60}})));
equation
  connect(weaDatInpCon.weaBus,pVModule_simple. weaBus) annotation (Line(
      points={{-60,70},{-40,70},{-40,54},{-10,54}},
      color={255,204,51},
      thickness=0.5));
  connect(pVModule_simple.scale,scale)  annotation (Line(points={{-12,46},{-40,
          46},{-40,0},{-120,0}},
                             color={0,0,127}));
  connect(WtokW.y,P_kw)
    annotation (Line(points={{41,10},{110,10}}, color={0,0,127}));
  connect(pVModule_simple.P,P)
    annotation (Line(points={{11,50},{110,50}}, color={0,0,127}));
  connect(WtokW.u,pVModule_simple. P) annotation (Line(points={{18,10},
          {14,10},{14,50},{11,50}}, color={0,0,127}));
  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    uses(
      SCooDER(version="6"),
      Buildings(version="6.0.0"),
      Modelica(version="3.2.2")));
end PV;
