name := "statnlpbook"

organization := "uk.ac.ucl.cs.mr"

organizationHomepage := Some(url("http://mr.cs.ucl.ac.uk"))

version := "1.0-SNAPSHOT"

scalaVersion := "2.11.4"

licenses := Seq("The Apache Software License, Version 2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.txt"))

homepage := Some(url("http://wolfe-pack.github.io/moro"))

resolvers ++= Seq(
  Resolver.file("Local repo", file(Path.userHome.absolutePath + "/.ivy2/local"))(Resolver.ivyStylePatterns),
  "IESL Release" at "https://dev-iesl.cs.umass.edu/nexus/content/groups/public",
  Resolver.mavenLocal,
  Resolver.defaultLocal,
  Resolver.sonatypeRepo("snapshots"),
  Resolver.sonatypeRepo("releases"),
  "Wolfe Release" at "http://homeniscient.cs.ucl.ac.uk:8081/nexus/content/repositories/releases",
  "Wolfe Snapshots" at "http://homeniscient.cs.ucl.ac.uk:8081/nexus/content/repositories/snapshots",
  "UIUC Releases" at "http://cogcomp.cs.illinois.edu/m2repo"
)

// disable using the Scala version in output paths and artifacts
crossPaths := false

credentials += Credentials(Path.userHome / ".ivy2" / ".credentials")

publishMavenStyle := true

publishTo := {
  val nexus = "https://dev-iesl.cs.umass.edu/nexus/"
  if (isSnapshot.value)
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases"  at nexus + "content/repositories/releases")
}

publishArtifact in Test := false

libraryDependencies ++= Seq(
  "net.sf.trove4j" % "trove4j" % "3.0.3",
  //"org.scalautils" % "scalautils_2.11" % "2.0",
  "org.scalatest" %% "scalatest" % "2.2.4" % "test",
  "org.sameersingh.htmlgen" % "htmlgen" % "0.3-SNAPSHOT",
  "org.sameersingh.scalaplot" % "scalaplot" % "0.1",
  "ml.wolfe" %% "wolfe-core" % "0.5.0-SNAPSHOT" exclude("org.slf4j", "slf4j-simple"),
  "ml.wolfe" %% "wolfe-util" % "0.5.0-SNAPSHOT" exclude("org.slf4j", "slf4j-simple"),
  "ml.wolfe" %% "wolfe-examples" % "0.5.0-SNAPSHOT" exclude("org.slf4j", "slf4j-simple"),
  "ml.wolfe" %% "wolfe-nlp" % "0.5.0-SNAPSHOT" exclude("org.slf4j", "slf4j-simple"),
  "ml.wolfe" %% "wolfe-ui" % "0.5.0-SNAPSHOT" exclude("org.slf4j", "slf4j-simple"),
  "org.scala-lang" % "scala-library" % "2.11.4"
)

